import torch
import numpy as np
from .base_zo_trainer import BaseZOTrainer
from .schedulers import hessian_smooth_scheduler

class HiZOOTrainer(BaseZOTrainer):
    """
    Implementation of HiZOO (Hessian-Guided Zeroth-Order).
    """
    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        # HiZOO state (Hessian Diagonal Estimate)
        self.hizoo_hessian = {}
        for name, param in self.named_parameters_to_optim:
            self.hizoo_hessian[name] = torch.ones_like(param.data)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        HiZOO Step:
        1. Estimates diagonal Hessian.
        2. Perturbs parameters scaled by inverse Hessian sqrt.
        """
        model.eval()
        self.zo_random_seed = np.random.randint(1000000000)
        
        # Need original loss for Hessian estimation
        loss_orig = self.zo_forward(model, inputs)
        
        # Perturb using Hessian scaling
        self._perturb_hizoo(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        self._perturb_hizoo(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        self._perturb_hizoo(scaling_factor=1) # Restore
        
        # Update Hessian Estimation
        self._update_hizoo_hessian(loss_orig, loss1, loss2)
        
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        self._update_hizoo()
        
        return loss1

    def _perturb_hizoo(self, scaling_factor):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
            # Scaling: z / sqrt(H)
            h_scale = torch.sqrt(self.hizoo_hessian[name]) + 1e-12
            param.data += (scaling_factor / h_scale) * z * self.args.zo_eps

    def _update_hizoo_hessian(self, loss_orig, loss1, loss2):
        """
        Estimate Hessian diagonal:
        H_est = |f(x+z) + f(x-z) - 2f(x)| / (2 * eps^2) * z^2
        """
        torch.manual_seed(self.zo_random_seed)
        smooth = hessian_smooth_scheduler(self.args.hessian_smooth_type, self.state.global_step, self.state.max_steps)
        
        # Calculate second order difference scalar
        diff = torch.abs(loss1 + loss2 - 2 * loss_orig).item()
        
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
            hessian_temp = self.hizoo_hessian[name] * (z**2)
            hessian_estimator = (diff * hessian_temp * smooth) / (2 * self.args.zo_eps**2)
            # EMA Update
            self.hizoo_hessian[name] = (1 - smooth) * self.hizoo_hessian[name] + hessian_estimator

    def _update_hizoo(self):
        torch.manual_seed(self.zo_random_seed)
        lr = self._get_learning_rate()
        
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
            h_scale = torch.sqrt(self.hizoo_hessian[name]) + 1e-12
            
            # Preconditioned gradient
            grad_est = (self.projected_grad * z) / h_scale
            
            if self.args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                grad_est += self.args.weight_decay * param.data
                
            param.data -= lr * grad_est