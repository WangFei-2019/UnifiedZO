import torch
import math
from .adalezo_trainer import AdaLeZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaZOAdaMUTrainer(AdaLeZOTrainer):
    """
    Implementation of AdaZOAdaMU (AdaLeZO + ZO-AdaMU).
    Combines Adaptive Layer-wise selection (AdaLeZO) with Adaptive Momentum Update (ZO-AdaMU).
    
    Logic:
    1. Select active layers using Bandit (AdaLeZO).
    2. Estimate gradients only for active layers.
    3. Apply IPW (Inverse Probability Weighting) to unbiasedly scale the gradients.
    4. Update parameters using Adam logic (First & Second moments) instead of SGD.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        
        # [ZO-AdaMU Logic] Initialize Adam-like states: m (exp_avg) and v (exp_avg_sq)
        # We maintain these states for ALL parameters to ensure continuity, 
        # even though we only update a subset at each step.
        self.exp_avg = {}
        self.exp_avg_sq = {}
        
        for name, param in self.named_parameters_to_optim:
            self.exp_avg[name] = torch.zeros_like(param.data)
            self.exp_avg_sq[name] = torch.zeros_like(param.data)

    def _update_adalezo(self):
        """
        Override: Update parameters using IPW + ZO-AdaMU (Adam) logic.
        This replaces the standard SGD-like update in AdaLeZOTrainer.
        """
        args = self.args
        lr = self._get_learning_rate()
        
        # ZO-AdaMU Hyperparameters
        beta1 = args.zo_adamu_beta1
        beta2 = args.zo_adamu_beta2
        eps = args.zo_adamu_epsilon
        
        # Bias Correction for Adam
        # global_step is usually 0-indexed, so +1
        step = self.state.global_step + 1
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Bandit Reward Signal (Magnitude of projected gradient)
        step_reward = abs(self.projected_grad.item())

        # Iterate ONLY over active layers (Sparse Update)
        for layer_key in self.current_active_layers:
            prob = self.current_layer_probs_map[layer_key]
            
            # --- IPW Calculation ---
            # Standard AdaLeZO IPW with clipping
            raw_ipw = 1.0 / (prob * len(self.current_active_layers) + 1e-8)
            ipw_weight = min(raw_ipw, args.adalezo_ipw_clip)
            
            # Note: We do NOT use adalezo_layer_momentum here because 
            # ZO-AdaMU already handles adaptive scaling via 'v' (second moment).
            scale_factor = ipw_weight

            # --- Parameter Update with Adam Logic ---
            # Must use SAME seed as _perturb_active_layers to reproduce 'z'
            torch.manual_seed(self.zo_random_seed + layer_key)
            
            for name, param in self.params_by_layer[layer_key]:
                # 1. Regenerate z
                z = self.generate_random_noise(
                    param.data.size(), 
                    param.data.device, 
                    param.data.dtype, 
                    args.perturb_type
                )
                
                # 2. Reconstruct Gradient: g = projected_grad * z * IPW_scale
                grad_est = self.projected_grad * z * scale_factor
                
                # 3. Apply Weight Decay (ZO-AdaMU style: added to gradient)
                if args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    grad_est += args.weight_decay * param.data
                
                # 4. Update Moments (Adam)
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                self.exp_avg[name].mul_(beta1).add_(grad_est, alpha=1 - beta1)
                
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                self.exp_avg_sq[name].mul_(beta2).addcmul_(grad_est, grad_est, value=1 - beta2)
                
                # 5. Compute Adaptive Step Size
                denom = (self.exp_avg_sq[name].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                
                # 6. Update Parameter: theta = theta - step_size * m / denom
                param.data.addcdiv_(self.exp_avg[name], denom, value=-step_size)

            # --- Update Bandit Stats (Same as AdaLeZO) ---
            idx = self.sorted_layer_keys.index(layer_key)
            self.layer_counts[idx] += 1
            self.layer_avg_rewards[idx] += (step_reward - self.layer_avg_rewards[idx]) / self.layer_counts[idx]