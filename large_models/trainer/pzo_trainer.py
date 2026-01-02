import torch
import numpy as np
import math
import logging
from collections import deque
from .base_zo_trainer import BaseZOTrainer

logger = logging.getLogger(__name__)

class PZOTrainer(BaseZOTrainer):
    """
    Implementation of PseuZO (Momentum/Sliding Window ZO).
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        # PseuZO state (Sliding Window & Momentum)
        self.sliding_window = deque(maxlen=self.args.sliding_window_length)
        self.coefficients = []
        self.momentum_fb_min = 0.0
        self.momentum_fb_max = 1.0 # Default value from original PseuZO code
        self.grad_last = None
        self.o_last = None
        
        # Init coefficients list
        self.reset_momentum_fb(self.momentum_fb_max)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        PseuZO Step:
        1. Calculate gradient w.r.t hidden states (grad_last) on unperturbed model.
        2. Perturb model.
        3. Calculate change in output states (o = o1 - o0).
        4. Store (seed, o) in sliding window.
        """
        model.eval()
        
        # --- 1. Dynamic Momentum Scheduling ---
        # Re-implemented based on original PseuZO logic
        self._update_momentum_coefficient()

        # --- 2. Get Gradient on Unperturbed model (and state o0) ---
        # need_grad=True to get grad_last
        # We rely on the wrapper 'forward_wrap_with_option_len_pzo' being active
        loss0, o0, grad_last = self.pzo_forward(model, inputs, need_grad=True)
        
        # Store gradient for update phase
        # Handling potential tuple return from DataParallel or specific models
        self.grad_last = grad_last[0] if isinstance(grad_last, tuple) else grad_last 

        # --- 3. Perturb ---
        seed = np.random.randint(1e9)
        self.zo_random_seed = seed # Save for update
        self._perturb_mezo(1) # Reuse basic perturbation logic (same as MeZO)
        
        # --- 4. Forward on perturbed model (only need state o1, no grad needed) ---
        loss1, o1, _ = self.pzo_forward(model, inputs, need_grad=False)
        
        # Restore model
        self._perturb_mezo(-1) # Restore
        
        # --- 5. Calculate Difference in Output Space ---
        # o0 and o1 are (bsz, seq_len, hidden_dim) or (bsz, seq_len, vocab)
        o_diff = o1 - o0
        
        # Store in window
        self.sliding_window.append((seed, o_diff))
        
        # --- 6. Update ---
        self._update_pzo()

        return loss1

    def _update_pzo(self):
        """
        PseuZO Update:
        Project the current gradient (grad_last) onto the history of output differences (o_diff).
        """
        lr = self._get_learning_rate()
        args = self.args
        
        dot_products = []
        seeds = []
        
        # Calculate dot products <grad_last, o_diff>
        for seed, o_diff in self.sliding_window:
            # Handle potential sequence length mismatch (truncation)

            if o_diff.shape[0] != self.grad_last.shape[0]:
                continue

            seq_o = o_diff.shape[1]
            seq_g = self.grad_last.shape[1]
            min_seq = min(seq_o, seq_g)
            
            if seq_o > min_seq:
                curr_o = o_diff[:, -min_seq:, :]
                curr_g = self.grad_last
            elif seq_g > min_seq:
                curr_o = o_diff
                curr_g = self.grad_last[:, -min_seq:, :]
            else:
                curr_o = o_diff
                curr_g = self.grad_last
                
            # Dot product over (Batch, Seq, Dim)
            dot = torch.sum(curr_o * curr_g, dim=(-3, -2, -1))
            dot_products.append(dot)
            seeds.append(seed)
            
        # Apply Coefficients (Momentum)
        # Using the last N coefficients for the items in window
        if len(dot_products) > 0:
            current_coeffs = self.coefficients[-len(dot_products):]
            
            final_weights = [co * dot.item() / self.args.zo_eps for co, dot in zip(current_coeffs, dot_products)]
            
            # Aggregate Parameter Updates
            with torch.no_grad():
                for i, (project_value, seed) in enumerate(zip(final_weights, seeds)):
                    torch.manual_seed(seed)

                    for name, param in self.named_parameters_to_optim:
                        z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)

                        if self.args.weight_decay > 0 and ("bias" not in name and "norm" not in name):
                            # Update rule with Weight Decay
                            param.data -= lr * (project_value * z + self.args.weight_decay * param.data)
                        else:
                            # Update rule without Weight Decay
                            param.data -= lr * (project_value * z)

        # Step LR Scheduler
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def pzo_forward(self, model, inputs, need_grad=False):
        """
        PZO forward for UnifiedZO compatibility.
        """
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            
            # Use inference_mode context or just call model
            # IMPORTANT: Pass is_pzo_step=True
            if need_grad:
                 outputs = model(need_grad=True, is_pzo_step=True, **inputs)
                 loss_val, state, grad_last = outputs.loss
            else:
                 outputs = model(need_grad=False, is_pzo_step=True, **inputs)
                 loss_val = outputs.loss
                 grad_last = None
                 
                 if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                     state = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
                 elif hasattr(outputs, 'logits') and outputs.logits is not None:
                     state = outputs.logits
                 else:
                     state = None

            if self.args.n_gpu > 1:
                loss_val = loss_val.mean()

        return loss_val, state, grad_last

    def _perturb_mezo(self, scaling_factor):
        # Re-using MeZO perturbation logic but without inheritance to keep files clean if needed,
        # or we could make MeZOTrainer utilities public. Here we just duplicate the simple loop.
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
            param.data += scaling_factor * z * self.args.zo_eps

    # --- Momentum Scheduling Logic (Ported from original PseuZO) ---

    def _update_momentum_coefficient(self):
        epoch = self.state.epoch if self.state.epoch is not None else 0
        num_train_epochs = self.state.num_train_epochs if self.state.num_train_epochs is not None else 1
        
        # Example scheduler: Cyclic Hyperbola
        def cyclic_hyperbola(t, T, k):
            cyc = T // k
            if t >= cyc * k:
                return self.momentum_fb_min
            t = t % cyc
            return (self.momentum_fb_max / (1 + 10 * t) if t <= 10 else self.momentum_fb_min)

        # Update momentum_fb based on schedule
        new_momentum = cyclic_hyperbola(epoch, num_train_epochs, 2)
        self.reset_momentum_fb(new_momentum)

    def reset_momentum_fb(self, momentum_fb):
        self.momentum_fb = momentum_fb
        self.coefficients = []
        for i in range(self.args.sliding_window_length):
            if i == 0:
                self.coefficients.append(1.0)
            else:
                self.coefficients = [co * self.momentum_fb for co in self.coefficients]
                self.coefficients.append(1.0)