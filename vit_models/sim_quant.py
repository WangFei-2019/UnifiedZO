import torch
import torch.nn as nn
import torch.nn.functional as F

class SimulatedQuantLinear(nn.Module):
    """
    Simulates a GPTQ-style Quantized Linear Layer.
    - Weights are quantized group-wise and FROZEN.
    - Scales are Group-wise Matrices (In//Group, Out) and TRAINABLE. 
      (Modified to match GPTQ layout: [Num_Groups, Out_Features])
    """
    def __init__(self, in_features, out_features, bias=True, bits=4, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        if in_features % group_size != 0:
            raise ValueError(f"in_features {in_features} must be divisible by group_size {group_size}")
        
        self.num_groups = in_features // group_size

        self.scales = nn.Parameter(torch.ones(self.num_groups, out_features, dtype=torch.float16))

        self.register_buffer('qweight_sim', torch.randn(out_features, in_features, dtype=torch.float16))

        # Bias (Optional)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

    @torch.no_grad()
    def import_from_linear(self, linear_layer):
        """
        Initialize from a standard nn.Linear layer.
        """
        weight = linear_layer.weight.data.to(dtype=torch.float16)
        
        # Reshape to [Out, Num_Groups, Group_Size]
        w_reshaped = weight.reshape(self.out_features, self.num_groups, self.group_size)

        # Calculate Scale per group
        # Shape: [Out, Num_Groups]
        max_val = torch.abs(w_reshaped).max(dim=2)[0]
        max_q = 2**(self.bits - 1) - 1
        scale = max_val / max_q
        scale = torch.clamp(scale, min=1e-5) 

        # Quantize Weight
        scale_expanded = scale.unsqueeze(-1)
        w_int = torch.round(w_reshaped / scale_expanded)
        w_int = torch.clamp(w_int, -max_q, max_q)
        
        self.scales.data.copy_(scale.t())
        
        self.qweight_sim.data.copy_(w_int.reshape(self.out_features, self.in_features))
        
        if linear_layer.bias is not None:
            self.bias.data.copy_(linear_layer.bias.data)
        
        self.to(linear_layer.weight.device)

    def forward(self, input):
        # W_eff = W_int * Scale
        
        # 1. Reshape weights: [Out, Groups, GroupSize]
        w_reshaped = self.qweight_sim.reshape(self.out_features, self.num_groups, self.group_size)
        
        # self.scales [Groups, Out] -> .t() -> [Out, Groups] -> .unsqueeze(-1) -> [Out, Groups, 1]
        s_reshaped = self.scales.t().unsqueeze(-1)
        
        # 3. Apply scales (Broadcasting: [Out, Groups, GroupSize] * [Out, Groups, 1])
        eff_weight_grouped = w_reshaped * s_reshaped
        
        # 4. Flatten back to [Out, In]
        eff_weight = eff_weight_grouped.reshape(self.out_features, self.in_features)

        if input.dtype != eff_weight.dtype:
            input = input.to(eff_weight.dtype)
        
        return F.linear(input, eff_weight, self.bias)

def replace_with_simulated_quant(model, bits=4, group_size=128, exclude_names=None):
    """
    Recursively replace nn.Linear with SimulatedQuantLinear, skipping excluded modules.
    """
    if exclude_names is None:
        exclude_names = ["classifier", "head"]

    for name, module in model.named_children():
        if name in exclude_names:
            print(f"Skipping excluded module (Keep FP16): {name}")
            continue

        if isinstance(module, nn.Linear):
            # Check feasibility (channel must be divisible by group_size)
            if module.in_features % group_size == 0:
                quant_layer = SimulatedQuantLinear(
                    module.in_features, 
                    module.out_features, 
                    bias=(module.bias is not None),
                    bits=bits,
                    group_size=group_size
                )
                quant_layer.import_from_linear(module)
                setattr(model, name, quant_layer)
            else:
                print(f"Skipping {name}: in_features {module.in_features} not divisible by {group_size}")
        else:
            replace_with_simulated_quant(module, bits, group_size, exclude_names)
    return model