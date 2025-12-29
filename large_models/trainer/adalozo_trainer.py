import torch
import numpy as np
from .adalezo_trainer import AdaLeZOTrainer

class AdaLoZOTrainer(AdaLeZOTrainer):
    """
    Implementation of AdaLoZO (AdaLeZO + LoZO).
    Combines Adaptive Layer-wise selection with Low-Rank perturbation.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        # Initialize LoZO state (V matrices)
        self.lozo_v = {}

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        AdaLoZO Step:
        1. Periodic LoZO V-matrix update.
        2. Standard AdaLeZO step (Select layers -> Perturb -> Forward -> ... -> Update).
        """
        # [LoZO Logic] Update V matrix periodically
        # This must happen before perturbation to ensure V exists and is up-to-date
        if self.state.global_step % self.args.lozo_step_interval == 0:
            self._update_lozo_v()

        # [AdaLeZO Logic] Call the parent's training step
        # The parent will call our overridden _perturb_active_layers and _update_adalezo methods
        return super().training_step(model, inputs, num_items_in_batch)

    def _update_lozo_v(self):
        """
        Generates new random V matrices for ALL 2D parameters.
        Copied logic from LoZOTrainer.
        """
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                # Shape assumption: [out_features, in_features]
                v = torch.randn(param.data.size(1), self.args.lozo_rank, 
                                device=param.data.device, dtype=param.data.dtype)
                self.lozo_v[name] = v

    def _perturb_active_layers(self, scaling_factor):
        """
        Override: Apply LoZO perturbation (U @ V.T) ONLY to active layers.
        """
        base_seed = self.zo_random_seed
        
        for layer_key in self.current_active_layers:
            # Deterministic seeding per layer
            torch.manual_seed(base_seed + layer_key)
            
            for name, param in self.params_by_layer[layer_key]:
                if param.data.ndim >= 2:
                    # [LoZO] Use Low-Rank Perturbation
                    if name not in self.lozo_v: self._update_lozo_v() # Safety check
                    v = self.lozo_v[name]
                    u = torch.randn(param.data.size(0), self.args.lozo_rank, 
                                    device=param.data.device, dtype=param.data.dtype)
                    perturbation = (u @ v.t())
                else:
                    # [Standard] Fallback for 1D params (bias/norm)
                    perturbation = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                
                # Apply perturbation
                param.data += scaling_factor * perturbation * self.args.zo_eps

    def _update_adalezo(self):
        """
        Override: Update parameters using IPW + LoZO Gradient estimation.
        """
        args = self.args
        lr = self._get_learning_rate()
        step_reward = abs(self.projected_grad)

        # Iterate only over active layers
        for layer_key in self.current_active_layers:
            prob = self.current_layer_probs_map[layer_key]
            count = self.current_layer_counts_map[layer_key]
            
            # --- IPW Calculation (Same as AdaLeZO) ---
            raw_ipw = 1.0 / (prob * self.num_active_draws + 1e-8)
            ipw_weight = min(raw_ipw, args.adalezo_ipw_clip)
            
            scale_factor = ipw_weight * count

            # Adaptive Layer Momentum logic (Optional)
            if args.adalezo_layer_momentum:
                idx = self.sorted_layer_keys.index(layer_key)
                layer_grad_est = self.projected_grad * ipw_weight
                current_energy = layer_grad_est ** 2
                self.layer_sq_grads[idx] = args.adalezo_beta * self.layer_sq_grads[idx] + (1 - args.adalezo_beta) * current_energy
                denom = torch.sqrt(self.layer_sq_grads[idx]) + 1e-8
                scale_factor = ipw_weight / denom

            # --- Parameter Update with LoZO Reconstruction ---
            # Must use SAME seed as _perturb_active_layers
            torch.manual_seed(self.zo_random_seed + layer_key)
            
            for name, param in self.params_by_layer[layer_key]:
                if param.data.ndim >= 2:
                    # [LoZO] Reconstruct Gradient
                    v = self.lozo_v[name]
                    u = torch.randn(param.data.size(0), self.args.lozo_rank, 
                                    device=param.data.device, dtype=param.data.dtype)
                    # Gradient Estimate = (U @ V.T) * projected_grad * IPW_scale
                    grad_est = (u @ v.t()) * self.projected_grad * scale_factor
                else:
                    # [Standard] Reconstruct Gradient
                    z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                    grad_est = z * self.projected_grad * scale_factor
                
                # Weight Decay
                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                         param.data.add_(grad_est + args.weight_decay * param.data, alpha=-lr)
                    else:
                         param.data.add_(grad_est, alpha=-lr)
                else:
                    param.data.add_(grad_est, alpha=-lr)

            # --- Update Bandit Stats (Same as AdaLeZO) ---
            idx = self.sorted_layer_keys.index(layer_key)
            self.layer_counts[idx] += count
            self.layer_avg_rewards[idx] += (step_reward - self.layer_avg_rewards[idx]) / self.layer_counts[idx]