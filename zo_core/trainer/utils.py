import torch
import math
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

def generate_random_noise(size, device, dtype, noise_type='Gaussian'):
    """
    Generates random noise tensor of specific shape and type.
    
    Args:
        size (torch.Size): The shape of the tensor.
        device (torch.device): Device to place the tensor on.
        dtype (torch.dtype): Data type.
        noise_type (str): Type of noise ('Gaussian', 'Rademacher', 'Third').
    
    Returns:
        torch.Tensor: The noise tensor.
    """
    if noise_type == 'Gaussian':
        # Standard Normal Distribution N(0, 1)
        return torch.normal(mean=0, std=1, size=size, device=device, dtype=dtype)
    
    elif noise_type == 'Rademacher':
        # Samples from {-1, 1} with equal probability
        return torch.randint(0, 2, size=size, device=device, dtype=dtype) * 2 - 1
    
    elif noise_type == 'Third':
        # Used in some specific ZO literature (e.g. STP)
        # Sample from a specific distribution involving sqrt(5)
        # Here we implement a simplified version or the specific one if needed.
        # Implementation based on the provided reference:
        p = (5 - math.sqrt(5)) / 10
        a = (1 + math.sqrt(5)) / 2
        b = (1 - math.sqrt(5)) / 2
        
        # Fast implementation using randint for thresholding
        threshold = int(p * 100000000)
        rand = torch.randint(0, 100000000, size, device=device)
        mask = rand < threshold
        out = torch.full(size, b, device=device, dtype=dtype)
        out[mask] = a
        return out
        
    else:
        raise NotImplementedError(f"Noise type {noise_type} not implemented.")


def random_gaussian_matrix(m, n, device, dtype):
    """
    Generates a random Gaussian matrix (used for LoZO U matrix).
    """
    return torch.randn(m, n, device=device, dtype=dtype)


def forward_wrap_with_option_len(
        self,
        input_ids=None,
        labels=None,
        option_len=None,
        num_options=None,
        return_dict=None,
        **kwargs
        ):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
      last option_len tokens 
    - num_options: a list of int indicating the number of options for each example (this will be #label
      words for classification tasks and #choices for multiple choice tasks), and a classification loss
      will be calculated.
    """
    outputs = self.original_forward(input_ids=input_ids, **kwargs)

    if labels is None:
        return outputs

    # in prompt tuning, we need to remove the virtual tokens from the logits to match the input ids
    logits = outputs.logits

    loss = None
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # Apply option len (do not calculate loss on the non-option part)
    # for _i, _len in enumerate(option_len):
    #     shift_labels[_i, :-_len] = -100
    # re-write the above code to avoid the for loop
    non_option_len = shift_labels.shape[1] - option_len
    mask = torch.arange(
        shift_labels.shape[1], device=shift_labels.device
    ).expand(shift_labels.shape[0], -1) < non_option_len.unsqueeze(-1)
    shift_labels[mask] = -100

    # Calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    
    if num_options is not None:
        # Train as a classification tasks
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100 # Option part
        shift_labels[~mask] = 0 # So that it doesn't mess up with indexing

        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) # (bsz x num_options, len)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1) # (bsz x num_options)

        if any([x != num_options[0] for x in num_options]):
            # Multi choice tasks with different number of options
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0) # (1, num_options)
                _labels = labels[start_id:end_id][0].unsqueeze(0) # (1)
                loss = loss_fct(_logits, _labels) + loss
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            num_options = num_options[0]
            selected_log_probs = selected_log_probs.view(-1, num_options) # (bsz, num_options)
            labels = labels.view(-1, num_options)[:, 0] # Labels repeat so we only take the first one
            loss = loss_fct(selected_log_probs, labels)

    else:
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def forward_wrap_with_option_len_pzo(self, need_grad=False, Hessian_estimate=False, adjust_lm_head=False, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=True, is_pzo_step=False, **kwargs):
    """
    PseuZO wrapper: Computes gradients w.r.t last_hidden_state.
    
    Args:
        is_pzo_step (bool): Flag to indicate if this forward pass is part of the PZO training loop.
                            If True, we preserve hidden states/logits for the optimizer.
                            If False (e.g. Eval), we drop them to save memory.
    """
    # 1. First forward pass (No Grad) to get hidden states
    with torch.no_grad():
        # Ensure output_hidden_states=True to capture the state for PZO
        outputs = self.original_forward(input_ids=input_ids, output_hidden_states=True, **kwargs)

        if labels is None:
            return outputs
        
        # Capture the last hidden state
        last_hidden_state = outputs.hidden_states[-1].detach() if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states.detach()
        del outputs # Free memory immediately

        # Prepare labels (masking logic)
        shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
        shift_labels[shift_labels == self.config.pad_token_id] = -100
        for _i, _len in enumerate(option_len):
            shift_labels[_i, :-_len] = -100

    # 2. Second partial forward pass (Enable Grad) from hidden state to loss
    with torch.enable_grad():
        last_hidden_state.requires_grad_(need_grad)
        loss = None
        loss_fct = CrossEntropyLoss(ignore_index=-100)

        # Re-compute logits from hidden state using lm_head
        logits = self.lm_head(last_hidden_state)

        # --- Classification or Causal LM Loss Calculation ---
        if num_options is not None:
            log_probs = F.log_softmax(logits[..., :-1, :].contiguous(), dim=-1)
            mask = shift_labels != -100
            shift_labels[~mask] = 0
            
            selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            selected_log_probs = (selected_log_probs * mask).sum(-1) / (mask.sum(-1) + 1e-9)

            if isinstance(num_options, list) and any([x != num_options[0] for x in num_options]):
                loss = 0
                start_id = 0
                count = 0
                while start_id < len(num_options):
                    end_id = start_id + num_options[start_id]
                    _logits = selected_log_probs[start_id:end_id].unsqueeze(0)
                    _labels = labels[start_id:end_id][0].unsqueeze(0)
                    loss = loss_fct(_logits, _labels) + loss
                    count += 1
                    start_id = end_id
                loss = loss / count
            else:
                n_opts = int(num_options[0]) if isinstance(num_options, list) or isinstance(num_options, torch.Tensor) else num_options
                selected_log_probs = selected_log_probs.view(-1, n_opts)
                labels = labels.view(-1, n_opts)[:, 0]
                loss = loss_fct(selected_log_probs, labels)
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    # 3. Handle Return Logic
    if need_grad:
        # [PZO Step 1]: Gradient calculation. Must return Tuple.
        if adjust_lm_head:
            grad_last = torch.autograd.grad(loss, [logits, self.lm_head.weight], create_graph=Hessian_estimate)
        else:
            grad_last = torch.autograd.grad(loss, logits if kwargs.get('logits',False) else last_hidden_state, create_graph=Hessian_estimate)[0]
        
        loss_export = (loss.detach(), last_hidden_state, grad_last)
        
        if not return_dict:
            output = (1,)
            return (loss_export,) + output

        return CausalLMOutputWithPast(loss=loss_export, logits=None)
    
    else:
        # [PZO Step 2 OR Eval]: No gradient needed.
        # Critical for Memory: Detach loss.
        final_loss = loss.detach()

        if is_pzo_step:
            # [PZO Step 2]: We need the state (hidden_state) for the algorithm.
            # We return it via hidden_states tuple.
            final_logits = None # We don't need logits for PZO (hidden version), save memory.
            return CausalLMOutputWithPast(
                loss=final_loss,
                logits=final_logits,
                hidden_states=(last_hidden_state,), # Pass state here
            )
        else:
            # [Standard Eval]: We need to save memory.
            # Drop logits and hidden_states to prevent OOM during accumulation.
            final_logits = None 
            return CausalLMOutputWithPast(loss=final_loss, logits=final_logits)


def forward_wrap_with_option_len_pzo_logits(self, need_grad=False, Hessian_estimate=False, adjust_lm_head=False, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=True, is_pzo_step=False, **kwargs):
    """
    PseuZO wrapper (Logits version).
    """
    # 1. Forward to get logits
    with torch.no_grad():
        outputs = self.original_forward(input_ids=input_ids, output_hidden_states=False, **kwargs)
        if labels is None: return outputs
        logits = outputs.logits.detach()
        del outputs
        
        shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
        shift_labels[shift_labels == self.config.pad_token_id] = -100
        for _i, _len in enumerate(option_len):
            shift_labels[_i, :-_len] = -100

    # 2. Compute Loss from detached logits
    with torch.enable_grad():
        loss = None
        logits.requires_grad_(need_grad)
        loss_fct = CrossEntropyLoss(ignore_index=-100)

        if num_options is not None:
            # Classification Logic
            log_probs = F.log_softmax(logits[..., :-1, :].contiguous(), dim=-1)
            mask = shift_labels != -100
            shift_labels[~mask] = 0
            selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            selected_log_probs = (selected_log_probs * mask).sum(-1) / (mask.sum(-1) + 1e-9)

            if isinstance(num_options, list) and any([x != num_options[0] for x in num_options]):
                loss = 0
                start_id = 0
                count = 0
                while start_id < len(num_options):
                    end_id = start_id + num_options[start_id]
                    _logits = selected_log_probs[start_id:end_id].unsqueeze(0)
                    _labels = labels[start_id:end_id][0].unsqueeze(0)
                    loss = loss_fct(_logits, _labels) + loss
                    count += 1
                    start_id = end_id
                loss = loss / count
            else:
                n_opts = int(num_options[0]) if isinstance(num_options, list) or isinstance(num_options, torch.Tensor) else num_options
                selected_log_probs = selected_log_probs.view(-1, n_opts)
                labels = labels.view(-1, n_opts)[:, 0]
                loss = loss_fct(selected_log_probs, labels)
        else:
            # Causal LM Logic
            shift_logits = logits[..., :-1, :].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    # 3. Handle Return Logic
    if need_grad:
        # [PZO Step 1]
        if adjust_lm_head:
             grad_last = torch.autograd.grad(loss, [logits, self.lm_head.weight], create_graph=Hessian_estimate)
        else:
            grad_last = torch.autograd.grad(loss, logits, create_graph=Hessian_estimate)[0]
            
        loss_export = (loss.detach(), logits, grad_last)
        if not return_dict: return (loss_export,) + (1,)
        return CausalLMOutputWithPast(loss=loss_export, logits=None)

    else:
        # [PZO Step 2 OR Eval]
        final_loss = loss.detach()
        
        if is_pzo_step:
            # [PZO Step 2]: For logits version, the 'state' IS the logits.
            # We must return it so PZO can compute o1 - o0.
            return CausalLMOutputWithPast(loss=final_loss, logits=logits)
        else:
            # [Standard Eval]: Drop logits to prevent OOM.
            return CausalLMOutputWithPast(loss=final_loss, logits=None)


def forward_wrap_with_option_len_fzoo(self, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=None, **kwargs):
    """
    Dedicated forward wrapper for FZOO to handle 'n' argument and potential state injection.
    """
    # [FZOO Specific Logic]: Pop 'n' to avoid TypeError in base model
    n = kwargs.pop("n", 1)
    
    # Optional: If your implementation of ParallelOPT/FZOO relies on self._custom_n, set it here.
    # self._custom_n = n 
    
    # Call original forward
    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    
    if labels is None:
        return outputs
        
    logits = outputs.logits
    loss = None

    # Shift logits and labels
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # Apply option_len masking
    if option_len is not None:
        for _i, _len in enumerate(option_len):
            shift_labels[_i, :-_len] = -100

    loss_fct = CrossEntropyLoss(ignore_index=-100)

    if num_options is not None: 
        # --- Classification Mode ---
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100 
        shift_labels[~mask] = 0 

        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) 
        selected_log_probs = (selected_log_probs * mask).sum(-1) / (mask.sum(-1) + 1e-9) 

        if isinstance(num_options, list) and any([x != num_options[0] for x in num_options]):
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0) 
                _labels = labels[start_id:end_id][0].unsqueeze(0) 
                loss = loss_fct(_logits, _labels) + loss
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            n_opts = int(num_options[0]) if isinstance(num_options, list) or isinstance(num_options, torch.Tensor) else num_options
            selected_log_probs = selected_log_probs.view(-1, n_opts)
            labels = labels.view(-1, n_opts)[:, 0]
            loss = loss_fct(selected_log_probs, labels)
    else:
        # --- Standard Causal LM Mode ---
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
