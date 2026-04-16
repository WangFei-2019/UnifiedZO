import torch
import numpy as np
import math
from .base_zo_trainer import BaseZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class ZOAdaMUTrainer(BaseZOTrainer):
    """
    Implementation of ZO-AdaMU (Zeroth-Order Adaptive Momentum Update).
    
    Logic:
    1. Estimate gradient g using zeroth-order difference.
    2. Update first moment (m) and second moment (v) estimates similar to Adam.
    3. Update parameters: theta = theta - lr * m / (sqrt(v) + eps).
    
    Note: This trainer consumes significantly more memory than MeZO because it maintains 
    optimizer states (m and v) for all trainable parameters.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        
        # Initialize Adam-like states: exp_avg (m) and exp_avg_sq (v)
        # These need to be on the same device as the parameters
        self.exp_avg = {}
        self.exp_avg_sq = {}
        
        for name, param in self.named_parameters_to_optim:
            self.exp_avg[name] = torch.zeros_like(param.data)
            self.exp_avg_sq[name] = torch.zeros_like(param.data)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        ZO-AdaMU Training Step:
        1. Sample z
        2. Compute projected gradient estimate (scalar part).
        3. Update moments and parameters.
        """
        model.eval()

        # 1. Sample Seed for consistent perturbation
        self.zo_random_seed = np.random.randint(1000000000)
        
        # 2. Forward pass with positive perturbation (+1)
        self._perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        # 3. Forward pass with negative perturbation (-2 from current state of +1 -> -1)
        self._perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        # 4. Calculate Projected Gradient Estimate (Scalar)
        # projected_grad approx directional derivative along z
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        # Restore parameters to original state (+1 from current state of -1 -> 0)
        self._perturb_parameters(scaling_factor=1)
        
        # 5. Update Parameters using Adam logic
        self._update_zoadamu()
        
        return loss1

    def _perturb_parameters(self, scaling_factor):
        """
        Apply perturbation to parameters: theta += scaling_factor * z * eps
        """
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(
                param.data.size(), 
                param.data.device, 
                param.data.dtype, 
                self.args.perturb_type
            )
            param.data += scaling_factor * z * self.args.zo_eps

    def _update_zoadamu(self):
        """
        Compute estimated gradient vector, update moments, and apply Adam step.
        """
        # Reset seed to regenerate the same z used in forward passes
        torch.manual_seed(self.zo_random_seed)
        
        lr = self._get_learning_rate()
        beta1 = self.args.zo_adamu_beta1
        beta2 = self.args.zo_adamu_beta2
        eps = self.args.zo_adamu_epsilon
        
        # Step used for bias correction (Adam logic)
        # Note: state.global_step is 0-indexed usually, so we use +1 for correction
        step = self.state.global_step + 1
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        for name, param in self.named_parameters_to_optim:
            # Regenerate z
            z = self.generate_random_noise(
                param.data.size(), 
                param.data.device, 
                param.data.dtype, 
                self.args.perturb_type
            )
            
            # Construct Estimated Gradient: g = projected_grad * z
            grad_est = self.projected_grad * z
            
            # Apply Weight Decay to the gradient (decoupled weight decay like AdamW is often preferred, 
            # but here we follow standard Adam logic applied to gradient)
            if self.args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                grad_est += self.args.weight_decay * param.data
            
            # --- Update Moments ---
            
            # Update First Moment (m)
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            self.exp_avg[name].mul_(beta1).add_(grad_est, alpha=1 - beta1)
            
            # Update Second Moment (v)
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            # Note: addcmul_ performs: input + value * tensor1 * tensor2
            self.exp_avg_sq[name].mul_(beta2).addcmul_(grad_est, grad_est, value=1 - beta2)
            
            # --- Compute Update ---
            
            # Bias Correction
            denom = (self.exp_avg_sq[name].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            
            # Apply Update: theta = theta - step_size * m / denom
            param.data.addcdiv_(self.exp_avg[name], denom, value=-step_size)