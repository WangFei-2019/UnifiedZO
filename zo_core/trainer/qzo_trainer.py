import torch
import numpy as np
from .base_zo_trainer import BaseZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class QZOTrainer(BaseZOTrainer):
    """
    Implementation of QZO (Quantized Zeroth-Order Optimization).
    Supports GPTQ and AQLM quantized models.
    """

    def __init__(self, model, args, **kwargs):
        _old_is_quantized = getattr(model, "is_quantized", False)
        if _old_is_quantized:
            model.is_quantized = False
        
        try:
            super().__init__(model, args, **kwargs)
        finally:
            # Restore the flag to ensure correct behavior elsewhere (e.g. saving)
            if _old_is_quantized:
                model.is_quantized = True

        self.fp16_to_optimize = {
            'scales': [], # list of params
            'regular': [] # list of (name, param)
        }
        self.int_to_optimize = {
            'qweight': [],
            'qzeros': []
        }
        
        # Momentum state
        self.fp16_to_optimize_momentum = {
            'scales': {}, # map name -> momentum buffer
            'regular': {} # map name -> momentum buffer
        }
        
        # Initialize optimization groups
        self._identify_quantized_params()
        
        # Initialize momentum buffers if needed
        if self.args.momentum:
            self._init_momentum_buffers()

        if hasattr(self.model.config, "quantization_config"):
            self.args.zoquantified_scale = 2 ** (4 - self.model.config.quantization_config.bits)
        else:
            self.args.zoquantified_scale = 2 ** (4 - self.args.quantized_bit) 
        
    def _identify_quantized_params(self):
        """
        Identify which parameters are quantization scales and which are regular float16 params.
        Supports GPTQ and AQLM.
        """
        if self.args.quant_method == 'gptq':
            try:
                from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear
                from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
                
                for name, module in self.model.named_modules():
                    if isinstance(module, (TritonV2QuantLinear, MarlinQuantLinear)):
                        self.fp16_to_optimize['scales'].append((name, module.scales))
                        self.int_to_optimize['qweight'].append(module.qweight)
                        self.int_to_optimize['qzeros'].append(module.qzeros)
            except ImportError:
                logger.warning("GPTQ modules not found. Ensure gptqmodel is installed.")

        elif self.args.quant_method == 'aqlm':
            try:
                from aqlm.inference import QuantizedLinear
                for name, module in self.model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        self.fp16_to_optimize['scales'].append((name, module.scales))
                        # Note: codebooks might be added here if supported
            except ImportError:
                logger.warning("AQLM modules not found. Ensure aqlm is installed.")
        
        else:
            found_sim_quant = False
            for name, module in self.model.named_modules():
                if module.__class__.__name__ == "SimulatedQuantLinear":
                    if hasattr(module, "scales") and isinstance(module.scales, torch.nn.Parameter):
                        self.fp16_to_optimize['scales'].append((name, module.scales))
                        found_sim_quant = True
            
            if found_sim_quant:
                logger.info(f"Successfully identified {len(self.fp16_to_optimize['scales'])} Simulated Quantized layers.")
        
        # Identify regular parameters (bias, norm, etc.)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fp16_to_optimize['regular'].append((name, param))

        if len(self.fp16_to_optimize['scales']) == 0:
            logger.warning("No quantized scales found to optimize! Check quant_method and model architecture.")

    def _init_momentum_buffers(self):
        for name, param in self.fp16_to_optimize['scales']:
            self.fp16_to_optimize_momentum['scales'][name] = 0
        for name, param in self.fp16_to_optimize['regular']:
            self.fp16_to_optimize_momentum['regular'][name] = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        QZO Step:
        1. Sample seed
        2. Perturb (+)
        3. Loss 1
        4. Perturb (-2) -> net effect (-)
        5. Loss 2
        6. Restore (+)
        7. Update
        """
        model.eval()

        self.zo_random_seed = np.random.randint(1000000000)
        
        # 1. Forward +
        self._perturb_qzo(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        # 2. Forward -
        self._perturb_qzo(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        # 3. Calculate Projected Gradient
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        # Restore
        self._perturb_qzo(scaling_factor=1)
        
        # 4. Update
        if self.args.momentum:
            self._update_qzo_momentum()
        else:
            self._update_qzo()
        
        return loss1

    def _perturb_qzo(self, scaling_factor):
        torch.manual_seed(self.zo_random_seed)
        
        # Perturb Regular Params (only if configured)
        if self.args.train_unquantized:
            for name, param in self.fp16_to_optimize['regular']:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data += scaling_factor * z * self.args.zo_eps
        
        # Perturb Scales (Always)
        for name, param in self.fp16_to_optimize['scales']:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # Note: args.zoquantified_scale used here as 'scale' in original code
            param.data += scaling_factor * z * self.args.zo_eps * self.args.zoquantified_scale

    def _update_qzo(self):
        torch.manual_seed(self.zo_random_seed)
        lr = self._get_learning_rate()
        
        # Clip Gradient
        if self.args.clip_zo_grad:
            self.projected_grad = min(max(-100, self.projected_grad), 100)

        # Update Regular Params
        if self.args.train_unquantized:
            for name, param in self.fp16_to_optimize['regular']:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                
                grad_est = self.projected_grad * z
                if self.args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    grad_est += self.args.weight_decay * param.data
                
                param.data -= lr * grad_est

        # Update Scales
        for name, param in self.fp16_to_optimize['scales']:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            
            # Note: Scale updates are often clamped to be positive if they represent std dev etc., 
            # but standard QZO code uses simple gradient descent with clamping for safety.
            # Original QZO: param.data = torch.clamp(param.data - lr * (grad * z) * scale, min=1e-7 * scale)
            
            update = lr * (self.projected_grad * z) * self.args.zoquantified_scale
            param.data = torch.clamp(param.data - update, min=1e-7 * self.args.zoquantified_scale)

    def _update_qzo_momentum(self):
        torch.manual_seed(self.zo_random_seed)
        lr = self._get_learning_rate()
        beta = self.args.beta
        
        if self.args.clip_zo_grad:
            self.projected_grad = min(max(-100, self.projected_grad), 100)

        # Regular Params
        if self.args.train_unquantized:
            for name, param in self.fp16_to_optimize['regular']:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                
                # Update Momentum
                self.fp16_to_optimize_momentum['regular'][name] = beta * self.fp16_to_optimize_momentum['regular'][name] + (1 - beta) * z * self.projected_grad
                
                grad_est = self.fp16_to_optimize_momentum['regular'][name]
                if self.args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    grad_est += self.args.weight_decay * param.data
                
                param.data -= lr * grad_est

        # Scales
        for name, param in self.fp16_to_optimize['scales']:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            
            self.fp16_to_optimize_momentum['scales'][name] = beta * self.fp16_to_optimize_momentum['scales'][name] + (1 - beta) * z * self.projected_grad
            
            update = lr * self.fp16_to_optimize_momentum['scales'][name] * self.args.zoquantified_scale
            param.data = torch.clamp(param.data - update, min=1e-7 * self.args.zoquantified_scale)