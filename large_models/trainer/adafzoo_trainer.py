import torch
import numpy as np
from .adalezo_trainer import AdaLeZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaFZooTrainer(AdaLeZOTrainer):
    """
    Implementation of AdaFZoo (AdaLeZO + FZOO).
    
    Combines:
    1. AdaLeZO: Layer-wise sparse selection using Multi-Armed Bandit.
    2. FZOO: Multiple perturbations (N) per step with variance reduction (normalization by std).
    
    Logic:
    1. Select active layers (Bandit).
    2. Perform N perturbations on these layers to collect N loss values.
    3. Estimate gradients using FZOO rule: (loss_i - baseline) / (N * std).
    4. Update parameters with IPW scaling.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        # FZOO Specific: Number of perturbations per step
        self.N = args.fzoo_n 

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        AdaFZoo Training Step.
        """
        model.eval()

        # 1. Baseline Loss (f(theta))
        # We need the unperturbed loss for the FZOO estimator: (f(theta+z) - f(theta))
        with torch.inference_mode():
            loss_baseline = self.zo_forward(model, inputs)
            loss_baseline_val = loss_baseline.detach().cpu().item()

        # 2. Collect N Perturbations
        losses = []
        seeds = []
        
        # We perform N perturbations on the *same* set of active layers
        for i in range(self.N):
            # Sample a unique seed for this perturbation
            seed = np.random.randint(1000000000)
            seeds.append(seed)
            self.zo_random_seed = seed
            
            # Perturb Active Layers (+1)
            self._perturb_active_layers(scaling_factor=1)
            
            # Forward
            loss = self.zo_forward(model, inputs)
            losses.append(loss.detach().cpu().item())
            
            # Restore Active Layers (Back to 0)
            # Since we added +1 * z * eps, subtracting same brings us back to original
            self._perturb_active_layers(scaling_factor=-1)

        # 3. Compute Statistics for FZOO Estimator
        losses_tensor = torch.tensor(losses, dtype=torch.float32)
        
        # Calculate standard deviation of the perturbed losses
        std = torch.std(losses_tensor, unbiased=False).item()
        if std == 0 or np.isnan(std):
            std = 1.0 # Avoid division by zero
            
        # Projected Gradients (Scalars) for each of the N perturbations
        # Formula: (Loss_i - Loss_baseline) / (N * std)
        # Note: We use N (perturbTimes) in the denominator as per FZOO paper/code
        projected_grads = (losses_tensor - loss_baseline_val) / (self.N * std)
        
        # 4. Update Parameters (Sparse + IPW + FZOO Aggregation)
        self._update_adafzoo(projected_grads, seeds)
        
        # 5. Bandit Re-sampling (AdaLeZO Logic)
        if self.state.global_step % self.args.adalezo_interval == 0:
            self._resample_layers()

        return loss_baseline

    def _update_adafzoo(self, projected_grads, seeds):
        """
        Update parameters using the aggregated estimator from N perturbations.
        """
        args = self.args
        lr = self._get_learning_rate()
        
        # Calculate a representative reward for the Bandit (e.g., mean magnitude of estimates)
        # This helps the bandit learn which layers are "sensitive"
        step_reward = torch.abs(projected_grads).mean().item()

        # Iterate only over active layers (Sparse Update)
        for layer_key in self.current_active_layers:
            prob = self.current_layer_probs_map[layer_key]
            
            # --- IPW Calculation ---
            raw_ipw = 1.0 / (prob * len(self.current_active_layers) + 1e-8)
            ipw_weight = min(raw_ipw, args.adalezo_ipw_clip)
            
            # Calculate the aggregated update for this layer across all N seeds
            # theta_new = theta - lr * IPW * sum_i ( projected_grad_i * z_i )
            
            for name, param in self.params_by_layer[layer_key]:
                # We need to accumulate the gradient estimate from all N seeds first
                total_grad_est = torch.zeros_like(param.data)
                
                for i, seed in enumerate(seeds):
                    # Regenerate noise z_i using the stored seed + layer_key
                    torch.manual_seed(seed + layer_key)
                    z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                    
                    # Accumulate: grad_est += scalar_i * z_i
                    total_grad_est += projected_grads[i] * z
                
                # Apply scaling
                total_grad_est *= ipw_weight
                
                # Update Parameter
                if args.weight_decay > 0:
                     if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                         param.data.add_(total_grad_est + args.weight_decay * param.data, alpha=-lr)
                     else:
                         param.data.add_(total_grad_est, alpha=-lr)
                else:
                    param.data.add_(total_grad_est, alpha=-lr)

            # --- Update Bandit Stats ---
            idx = self.sorted_layer_keys.index(layer_key)
            self.layer_counts[idx] += 1
            self.layer_avg_rewards[idx] += (step_reward - self.layer_avg_rewards[idx]) / self.layer_counts[idx]