# prompt_tuning.py
import logging
import warnings
from functools import partial
from typing import Optional, Callable, Union

import torch
from torch import nn
from transformers import PreTrainedModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PromptEmbedding(nn.Module):
    """
    The embedding layer for virtual prompt tokens.
    """
    def __init__(
            self,
            num_virtual_tokens: int,
            token_dim: int,
            init_by_real_text: bool,
            word_embeddings: Optional[nn.Module] = None,
            vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens

        self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
        
        # Initialize virtual tokens using real word embeddings if requested
        if init_by_real_text:
            logger.info(f"Initializing {num_virtual_tokens} virtual tokens with random real word embeddings.")
            if word_embeddings is None or vocab_size is None:
                raise ValueError("word_embeddings and vocab_size must be provided for real text initialization.")
                
            init_token_ids = torch.randint(
                low=0, high=vocab_size,
                size=(num_virtual_tokens,), dtype=torch.long
            ).to(word_embeddings.weight.device)

            with torch.no_grad():
                word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
                word_embedding_weights = word_embedding_weights.to(torch.float32)
                self.embedding.weight = nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # Return the learned embeddings for the prompt indices
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings


def _get_batch_size(input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]) -> int:
    """Helper to determine batch size from inputs."""
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


def _model_forward_hook(
        self,
        embedding_module: Callable,
        embedding_module_device_refer,
        hide_virtual_token_logits: bool,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
):
    """
    A wrapper for the model's forward pass. 
    It intercepts inputs, prepends the learned prompt embeddings, adjusts attention masks/labels, 
    and then calls the original model forward.
    """
    batch_size = _get_batch_size(input_ids, inputs_embeds)
    num_virtual_tokens = self.prompt_encoder.num_virtual_tokens
    
    # 1. Extend Attention Mask
    if attention_mask is not None:
        # Create attention mask for prompts (all 1s) and concat
        prefix_attention_mask = torch.ones(batch_size, num_virtual_tokens).to(attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
    
    # Position IDs are usually recalculated by the model automatically when None
    if kwargs.get("position_ids", None) is not None:
        warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        kwargs["position_ids"] = None
        
    kwargs.update(
        {
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
    )

    # 2. Extend Labels (Pad with -100 so prompt tokens are ignored in loss)
    if labels is not None:
        if len(labels.shape) == 1:
            # Sequence classification (one label per sequence) - no padding needed usually
            kwargs["labels"] = labels
        elif len(labels.shape) == 2:
            # Language modeling (token-wise labels) - pad prefix with -100
            kwargs["labels"] = torch.cat(
                (
                    -100 * torch.ones(batch_size, num_virtual_tokens).to(labels.device).long(),
                    labels,
                ),
                dim=1,
            )
        else:
            raise NotImplementedError("Not implemented for labels with shape {}".format(labels.shape))

    # 3. Extend Token Type IDs
    if kwargs.get("token_type_ids", None) is not None:
        kwargs["token_type_ids"] = torch.cat(
            (
                torch.zeros(batch_size, num_virtual_tokens).to(kwargs["token_type_ids"].device),
                kwargs["token_type_ids"],
            ),
            dim=1,
        ).long()

    # 4. Handle Mask Position (if applicable)
    if kwargs.get("mask_pos", None) is not None:
        kwargs["mask_pos"] = num_virtual_tokens + kwargs["mask_pos"]

    # 5. Construct Input Embeddings
    input_device = input_ids.device if input_ids is not None else inputs_embeds.device
    if inputs_embeds is None:
        # Get original embeddings for the input text
        inputs_embeds = embedding_module(input_ids.to(embedding_module_device_refer.device))
        inputs_embeds = inputs_embeds.to(input_device)
        
    # Get virtual prompt embeddings
    prompts = torch.arange(num_virtual_tokens).unsqueeze(0).expand(batch_size, -1).to(
        self.prompt_encoder.embedding.weight.device)
    prompts = self.prompt_encoder(prompts).to(dtype=inputs_embeds.dtype, device=input_device)

    # Concatenate: [Prompt Embeddings, Text Embeddings]
    inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

    # 6. Call Original Forward
    outputs = self.prompt_tuning_original_forward(inputs_embeds=inputs_embeds, **kwargs)
    
    # 7. Hide Virtual Token Logits (optional)
    if hide_virtual_token_logits and hasattr(outputs, "logits"):
        outputs.logits = outputs.logits[..., num_virtual_tokens:, :]
        
    return outputs


class PromptTuning:
    """
    Main class to inject Prompt Tuning into a PreTrainedModel.
    """

    def __init__(
            self,
            model: PreTrainedModel,
            num_virtual_tokens: int,
            init_by_real_tokens: Optional[bool] = False,
            hide_virtual_token_logits: Optional[bool] = True,
    ):
        """
        Prompt tuning model initializer.

        Parameters
        ----------
        model: PreTrainedModel
            The model to be tuned.
        num_virtual_tokens: int
            The number of virtual tokens to be added.
        init_by_real_tokens: bool
            Whether to initialize the virtual tokens by real tokens.
        hide_virtual_token_logits: bool
            Whether to remove logits corresponding to virtual tokens from output.
        """
        self.num_virtual_tokens = num_virtual_tokens
        hidden_dim = model.config.hidden_size

        # Identify the embedding module for different architectures
        if model.config.model_type == "opt":
            embedding_module = model.get_input_embeddings()
            embedding_module_device_refer = embedding_module.weight
        elif model.config.model_type == "roberta":
            if hasattr(model, "roberta"):
                embedding_module = partial(model.roberta.embeddings, past_key_values_length=num_virtual_tokens)
                embedding_module_device_refer = model.roberta.embeddings.word_embeddings.weight
            elif hasattr(model, "embeddings"):
                embedding_module = partial(model.embeddings, past_key_values_length=num_virtual_tokens)
                embedding_module_device_refer = model.embeddings.word_embeddings.weight
            else:
                raise ValueError(f"Cannot find embedding module in {model.__class__.__name__}")
        elif model.config.model_type in ["llama", "mistral", "gemma"]:
            embedding_module = model.get_input_embeddings()
            embedding_module_device_refer = embedding_module.weight
        else:
            # Fallback for generic models
            try:
                embedding_module = model.get_input_embeddings()
                embedding_module_device_refer = embedding_module.weight
            except AttributeError:
                raise NotImplementedError(f"Prompt Tuning not explicitly implemented for {model.config.model_type}")

        # Initialize the Prompt Encoder
        model.prompt_encoder = PromptEmbedding(
            num_virtual_tokens, hidden_dim, init_by_real_tokens,
            model.get_input_embeddings(), model.config.vocab_size
        )
        
        # Save original forward pass
        model.prompt_tuning_original_forward = model.forward

        if not hasattr(embedding_module_device_refer, "device"):
            raise ValueError(f"Cannot find device attribute in {embedding_module_device_refer.__class__.__name__}")

        # Replace forward method with the hook
        forward_hook_kwargs = {
            "embedding_module": embedding_module,
            "embedding_module_device_refer": embedding_module_device_refer,
            "hide_virtual_token_logits": hide_virtual_token_logits,
        }
        model.forward = partial(
            _model_forward_hook.__get__(model, type(model)),
            **forward_hook_kwargs
        )

        # Freeze all parameters except the prompt encoder
        logger.info("Freezing model parameters. Only 'prompt_encoder' will remain trainable.")
        for n, p in model.named_parameters():
            if "prompt_encoder" not in n:
                p.requires_grad = False