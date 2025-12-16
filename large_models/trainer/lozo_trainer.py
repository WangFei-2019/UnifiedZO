import torch
import numpy as np
from .base_zo_trainer import BaseZOTrainer

class LoZOTrainer(BaseZOTrainer):
    """
    Implementation of LoZO (Low-Rank Zeroth-Order Optimization).
    """
    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        # LoZO state (Low-Rank V matrices)
        self.lozo_v = {}

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        LoZO Step:
        Updates V matrix periodically.
        Perturbs using U @ V.T instead of full z.
        """
        model.eval()

        # Update V matrix if interval is met
        if self.state.global_step % self.args.lozo_step_interval == 0:
            self._update_lozo_v()

        self.zo_random_seed = np.random.randint(1000000000)
        
        self._perturb_lozo(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        self._perturb_lozo(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        self._perturb_lozo(scaling_factor=1) # Restore
        
        self.projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps)
        
        self._update_lozo()

        return loss1

    def _update_lozo_v(self):
        # Generate new random V matrices for low-rank layers
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                # Assuming shape [out_features, in_features]
                v = torch.randn(param.data.size(1), self.args.lozo_rank, 
                                device=param.data.device, dtype=param.data.dtype)
                self.lozo_v[name] = v

    def _perturb_lozo(self, scaling_factor):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                # Low-Rank: U @ V.T
                v = self.lozo_v[name]
                u = torch.randn(param.data.size(0), self.args.lozo_rank, 
                                device=param.data.device, dtype=param.data.dtype)
                perturbation = (u @ v.t())
            else:
                # Fallback to standard noise for 1D params (bias/layernorm)
                perturbation = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
            
            param.data += scaling_factor * perturbation * self.args.zo_eps

    def _update_lozo(self):
        torch.manual_seed(self.zo_random_seed)
        lr = self._get_learning_rate()
        
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                v = self.lozo_v[name]
                u = torch.randn(param.data.size(0), self.args.lozo_rank, 
                                device=param.data.device, dtype=param.data.dtype) 
                grad_est = (u @ v.t()) * self.projected_grad # / self.args.lozo_rank # Unlike the original LOZO code, which is biased, here we divide by rank to remove the bias. 
            else:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                grad_est = z * self.projected_grad

            if self.args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                grad_est += self.args.weight_decay * param.data
            
            param.data -= lr * grad_est