import torch
import numpy as np
from .adalezo_trainer import AdaLeZOTrainer
from .schedulers import hessian_smooth_scheduler
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaHiZOOTrainer(AdaLeZOTrainer):
    """
    Implementation of AdaHiZOO (AdaLeZO + HiZOO).
    Combines Adaptive Layer-wise selection with Hessian-guided perturbation/update.
    
    Logic:
    1. Select active layers (AdaLeZO).
    2. Compute original loss (L0) + perturbed losses (L1, L2).
    3. Estimate Hessian diagonal (H) only for active layers using finite difference (L1+L2-2L0).
    4. Apply updates preconditioned by sqrt(H), scaled by IPW.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        
        # [HiZOO Logic] Initialize Hessian Estimates
        # We maintain Hessian states for ALL parameters to ensure that
        # when a layer becomes active, it has a valid initialization (ones).
        self.hizoo_hessian = {}
        for name, param in self.named_parameters_to_optim:
            self.hizoo_hessian[name] = torch.ones_like(param.data)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        AdaHiZOO Step:
        Requires 3 forward passes (Original, Pos Perturb, Neg Perturb) to estimate Hessian.
        """
        model.eval()
        self.zo_random_seed = np.random.randint(1000000000)
        
        # 1. Forward pass (Original) - Needed for Hessian estimation
        loss_orig = self.zo_forward(model, inputs)
        
        # 2. Perturb Active Layers (+)
        self._perturb_active_layers(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        # 3. Perturb Active Layers (-)
        self._perturb_active_layers(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        # Restore
        self._perturb_active_layers(scaling_factor=1)
        
        # 4. Update Hessian Estimate (Only for active layers)
        self._update_hizoo_hessian(loss_orig, loss1, loss2)
        
        # 5. Projected Gradient
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        # 6. Update Parameters (IPW + Hessian scaling)
        self._update_adalezo()
        
        # 7. Resample Layers (AdaLeZO Logic)
        if self.state.global_step % self.args.adalezo_interval == 0:
            self._resample_layers()

        return loss1

    def _perturb_active_layers(self, scaling_factor):
        """
        Override: Apply HiZOO perturbation (z / sqrt(H)) ONLY to active layers.
        """
        base_seed = self.zo_random_seed
        
        for layer_key in self.current_active_layers:
            torch.manual_seed(base_seed + layer_key)
            
            for name, param in self.params_by_layer[layer_key]:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                
                # [HiZOO] Scaling: z / sqrt(H)
                h_scale = torch.sqrt(self.hizoo_hessian[name]) + 1e-12
                param.data += (scaling_factor / h_scale) * z * self.args.zo_eps

    def _update_hizoo_hessian(self, loss_orig, loss1, loss2):
        """
        Update Hessian diagonal estimates for active layers.
        H_est = |L1 + L2 - 2L0| / (2 * eps^2) * z^2
        """
        # Get smoothing factor from scheduler
        smooth = hessian_smooth_scheduler(self.args.hessian_smooth_type, self.state.global_step, self.state.max_steps)
        
        # Second order difference scalar
        second_order_diff = torch.abs(loss1 + loss2 - 2 * loss_orig).item()
        
        base_seed = self.zo_random_seed
        
        for layer_key in self.current_active_layers:
            torch.manual_seed(base_seed + layer_key)
            
            for name, param in self.params_by_layer[layer_key]:
                # Regenerate z
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                
                # Hessian Estimate
                hessian_temp = self.hizoo_hessian[name] * (z**2)
                hessian_estimator = (second_order_diff * hessian_temp * smooth) / (2 * self.args.zo_eps**2)
                
                # EMA Update
                self.hizoo_hessian[name] = (1 - smooth) * self.hizoo_hessian[name] + hessian_estimator

    def _update_adalezo(self):
        """
        Override: Update parameters using IPW + HiZOO Gradient estimation.
        grad = (pg * z / sqrt(H)) * IPW
        """
        args = self.args
        lr = self._get_learning_rate()
        step_reward = abs(self.projected_grad)

        for layer_key in self.current_active_layers:
            prob = self.current_layer_probs_map[layer_key]
            count = self.current_layer_counts_map[layer_key]
            
            # --- IPW Calculation ---
            raw_ipw = 1.0 / (prob * self.num_active_draws + 1e-8)
            ipw_weight = min(raw_ipw, args.adalezo_ipw_clip)
            
            # Note: HiZOO uses Hessian for adaptive scaling, so we typically 
            # disable 'adalezo_layer_momentum' to avoid double scaling.
            scale_factor = ipw_weight * count

            torch.manual_seed(self.zo_random_seed + layer_key)
            
            for name, param in self.params_by_layer[layer_key]:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                
                # [HiZOO] Preconditioned Gradient
                h_scale = torch.sqrt(self.hizoo_hessian[name]) + 1e-12
                grad_est = (self.projected_grad * z / h_scale) * scale_factor
                
                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        param.data.add_(grad_est + args.weight_decay * param.data, alpha=-lr)
                    else:
                        param.data.add_(grad_est, alpha=-lr)
                else:
                    param.data.add_(grad_est, alpha=-lr)

            # --- Update Bandit Stats ---
            idx = self.sorted_layer_keys.index(layer_key)
            self.layer_counts[idx] += count
            self.layer_avg_rewards[idx] += (step_reward - self.layer_avg_rewards[idx]) / self.layer_counts[idx]