import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SimulatedQuantLinear(nn.Module):
    """
    用於 VLM 零階優化的模擬量化線性層。
    """
    def __init__(self, in_features, out_features, bias=True, bits=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.scales = nn.Parameter(torch.ones(out_features, 1), requires_grad=True)
        
    def init_quant_params(self):
        max_val = self.weight.abs().max(dim=1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        self.scales.data = max_val / self.qmax

    def fake_quantize(self):
        w_q = torch.round(self.weight / self.scales)
        w_q = torch.clamp(w_q, self.qmin, self.qmax)
        w_dq = w_q * self.scales
        return w_dq

    def forward(self, input):
        w_dq = self.fake_quantize()
        return F.linear(input, w_dq, self.bias)

def apply_vlm_simulated_quantization(model, bits=4, quantize_llm=True, quantize_vision=False):
    replaced_count = 0
    target_modules = []
    
    # 1. 判断是否量化 LLM
    if quantize_llm and hasattr(model, "language_model"):
        target_modules.append(("LLM Backbone", model.language_model))
        logger.info("Targeting LLM Backbone for simulated quantization.")
        
    # 2. 判断是否量化 Vision Tower
    if quantize_vision and hasattr(model, "vision_tower"):
        target_modules.append(("Vision Tower", model.vision_tower))
        logger.info("Targeting Vision Tower for simulated quantization.")
        
    if not target_modules:
        logger.warning("No modules selected for quantization or attributes not found! Checking fallback...")
        target_modules.append(("Whole Model Fallback", model))

    # 3. 遍历目标模块注入 FakeQuant 层
    for module_desc, target_module in target_modules:
        module_replaced = 0
        for name, module in target_module.named_modules():
            # 避开预测头和视觉映射层
            if any(skip_name in name for skip_name in ["lm_head", "multi_modal_projector"]):
                continue
                
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    device = child_module.weight.device
                    dtype = child_module.weight.dtype
                    
                    quant_layer = SimulatedQuantLinear(
                        child_module.in_features,
                        child_module.out_features,
                        bias=(child_module.bias is not None),
                        bits=bits
                    )
                    quant_layer.weight.data.copy_(child_module.weight.data)
                    if child_module.bias is not None:
                        quant_layer.bias.data.copy_(child_module.bias.data)
                        
                    quant_layer.init_quant_params()
                    quant_layer.to(device=device, dtype=dtype)
                    
                    setattr(module, child_name, quant_layer)
                    module_replaced += 1
                    replaced_count += 1
        logger.info(f"[{module_desc}] Injected {module_replaced} QuantLinear layers.")
                    
    logger.info(f"Total successfully injected {replaced_count} SimulatedQuantLinear layers (bits={bits}).")
    return model