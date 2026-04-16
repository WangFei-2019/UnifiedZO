import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SimulatedQuantLinear(nn.Module):
    """
    Simulated Group-wise Quantized Linear Layer for VLM.
    Aligned with GPTQ/ViT structure to seamlessly support Low-Rank ZO (LQZO).
    - Weights are frozen.
    - Scales are Group-wise Matrices: [Num_Groups, Out_Features] and TRAINABLE.
    """
    def __init__(self, in_features, out_features, bias=True, bits=4, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        # Architecture Guard: Ensure clean grouping
        if in_features % group_size != 0:
            raise ValueError(f"in_features {in_features} must be divisible by group_size {group_size}")
            
        self.num_groups = in_features // group_size
        
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1
        
        # Frozen weight buffer [Out, In]
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
            
        # Trainable scales: Shape [Num_Groups, Out] - Matched exactly with GPTQ/LQZO expectations
        self.scales = nn.Parameter(torch.ones(self.num_groups, out_features), requires_grad=True)
        
    def init_quant_params(self):
        # 1. Reshape weight to [Out, Num_Groups, Group_Size]
        w_reshaped = self.weight.data.reshape(self.out_features, self.num_groups, self.group_size)
        
        # 2. Max value per group. Shape: [Out, Num_Groups]
        max_val = w_reshaped.abs().max(dim=2)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        
        # 3. Calculate scale
        scale = max_val / self.qmax
        
        # 4. Store transposed scale to match target shape [Num_Groups, Out]
        self.scales.data.copy_(scale.t())

    def fake_quantize(self):
        # 1. Reshape weight to [Out, Num_Groups, Group_Size]
        w_reshaped = self.weight.reshape(self.out_features, self.num_groups, self.group_size)
        
        # 2. Prepare scales. self.scales is [Num_Groups, Out]
        # Transpose to [Out, Num_Groups], then unsqueeze to [Out, Num_Groups, 1] for broadcasting
        safe_scales = torch.clamp(self.scales, min=1e-7)
        s_reshaped = safe_scales.t().unsqueeze(-1)
        
        # 3. Simulate Quantization (Fake Quant)
        w_q = torch.round(w_reshaped / s_reshaped)
        w_q = torch.clamp(w_q, self.qmin, self.qmax)
        
        # 4. Dequantize
        w_dq_grouped = w_q * s_reshaped
        
        # 5. Flatten back to [Out, In]
        w_dq = w_dq_grouped.reshape(self.out_features, self.in_features)
        return w_dq

    def forward(self, input):
        w_dq = self.fake_quantize()
        return F.linear(input, w_dq, self.bias)

def _replace_linear_with_quant_recursive(module: nn.Module, bits: int, group_size: int, prefix: str = "") -> int:
    """
    Recursively replaces nn.Linear with SimulatedQuantLinear safely.
    Skips layers where in_features is not divisible by group_size.
    """
    replaced_count = 0
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Avoid perturbing the language model head or multimodal projector
        if any(skip_name in full_name for skip_name in ["lm_head", "multi_modal_projector"]):
            continue
            
        if isinstance(child, nn.Linear):
            # Strict Divisibility Check
            if child.in_features % group_size == 0:
                quant_layer = SimulatedQuantLinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    bits=bits,
                    group_size=group_size
                )
                quant_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    quant_layer.bias.data.copy_(child.bias.data)
                    
                quant_layer.init_quant_params()
                quant_layer.to(device=child.weight.device, dtype=child.weight.dtype)
                
                setattr(module, name, quant_layer)
                replaced_count += 1
            else:
                logger.warning(f"Architecture Guard: Skipping {full_name} because in_features ({child.in_features}) is not divisible by group_size ({group_size}). Kept in FP16.")
        else:
            replaced_count += _replace_linear_with_quant_recursive(child, bits, group_size, full_name)
            
    return replaced_count

def apply_vlm_simulated_quantization(model, bits=4, group_size=128, quantize_llm=True, quantize_vision=False):
    total_replaced = 0
    
    if quantize_llm and hasattr(model, "language_model"):
        logger.info(f"Applying Grouped Quantization (group_size={group_size}) to LLM Backbone...")
        replaced = _replace_linear_with_quant_recursive(model.language_model, bits, group_size, "language_model")
        total_replaced += replaced
        
    if quantize_vision and hasattr(model, "vision_tower"):
        logger.info(f"Applying Grouped Quantization (group_size={group_size}) to Vision Tower...")
        replaced = _replace_linear_with_quant_recursive(model.vision_tower, bits, group_size, "vision_tower")
        total_replaced += replaced
        
    logger.info(f"Successfully injected {total_replaced} Group-wise SimulatedQuantLinear layers (bits={bits}).")
    return model