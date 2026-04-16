import torch
import copy
import numpy as np
from .adalezo_trainer import AdaLeZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaMeZOSVRGTrainer(AdaLeZOTrainer):
    """
    Implementation of AdaMeZO-SVRG.
    Combines AdaLeZO (Adaptive Layer-wise Sparse Update) with MeZO-SVRG (Variance Reduction).
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        
        # --- SVRG Hyperparameters ---
        self.svrg_q = args.svrg_q # Anchor update frequency
        self.svrg_k = args.svrg_k # Samples for anchor gradient estimation
        
        # --- SVRG State Storage ---
        # Anchor params stored on CPU to save VRAM
        self.anchor_params = {} 
        self.anchor_mu = {}     # The "full" gradient at the anchor point
        self.has_initialized_anchor = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        AdaMeZO-SVRG Step:
        1. Periodic Anchor Update (Global).
        2. Sparse Inner Step (Selected Layers only) with SVRG correction.
        3. Periodic Bandit Resampling (AdaLeZO logic).
        """
        model.eval()

        # --- 1. Periodic Anchor Update (Outer Loop) ---
        # Update anchor if first step OR interval reached
        if not self.has_initialized_anchor or (self.state.global_step > 0 and self.state.global_step % self.svrg_q == 0):
            self._update_anchor(model, inputs)
            self.has_initialized_anchor = True

        # --- 2. Adaptive Sparse Step (Inner Loop) ---
        loss = self._ada_svrg_step(model, inputs)

        # --- 3. Periodic Bandit Resampling (AdaLeZO Logic) ---
        # Important: Since we override training_step, we must manually trigger the resampling
        if self.state.global_step % self.args.adalezo_interval == 0:
            self._resample_layers()
            # Log probabilities if the method exists (for debugging)
            if hasattr(self, '_log_probs'):
                self._log_probs()

        return loss

    def _update_anchor(self, model, inputs):
        """
        Updates the anchor model (\\tilde{w}) and calculates the reference gradient (\\mu).
        Note: We compute \\mu for ALL parameters because any layer might be selected 
        by the Bandit in future steps.
        """
        logger.info(f"[AdaMeZO-SVRG] Updating anchor at step {self.state.global_step}...")
        
        # Save current parameters as Anchor (CPU)
        self.anchor_params = {}
        # Iterate over all parameter groups
        for layer_key, param_list in self.params_by_layer.items():
            for name, param in param_list:
                self.anchor_params[name] = param.detach().cpu().clone()
        
        # Estimate "Full" Gradient (Mu) at Anchor
        self.anchor_mu = {}
        # Initialize zero tensors for accumulation
        for layer_key, param_list in self.params_by_layer.items():
            for name, param in param_list:
                self.anchor_mu[name] = torch.zeros_like(param)
        
        scale = 1.0 / self.svrg_k
        
        # Use simple global perturbation for anchor gradient estimate
        for _ in range(self.svrg_k):
            seed = np.random.randint(1000000000)
            
            # Forward +
            self._apply_global_noise(model, seed, scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)
            
            # Forward -
            self._apply_global_noise(model, seed, scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)
            
            # Restore
            self._apply_global_noise(model, seed, scaling_factor=1)
            
            # Grad estimate
            projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
            
            # Accumulate into Mu
            torch.manual_seed(seed)
            for layer_key, param_list in self.params_by_layer.items():
                for name, param in param_list:
                    z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
                    self.anchor_mu[name] += (projected_grad * z * scale).detach()
        
        logger.info(f"[AdaMeZO-SVRG] Anchor updated.")

    def _ada_svrg_step(self, model, inputs):
        """
        Performs the sparse update with variance reduction.
        """
        # 1. Use Active Layers directly from AdaLeZO state
        active_layers = self.current_active_layers
        probs_map = self.current_layer_probs_map
        
        if not active_layers:
            # Should essentially not happen if k_ratio > 0
            return torch.tensor(0.0, device=self.args.device)

        # 2. Sample Seed for this step
        step_seed = np.random.randint(1000000000)
        
        # --- A. Compute Gradient at Current Model (w_t) for Active Layers ---
        # Apply noise ONLY to active layers
        self._perturb_active_layers(scaling_factor=1, seed=step_seed)
        loss_curr_1 = self.zo_forward(model, inputs)
        
        self._perturb_active_layers(scaling_factor=-2, seed=step_seed)
        loss_curr_2 = self.zo_forward(model, inputs)
        
        self._perturb_active_layers(scaling_factor=1, seed=step_seed) # Restore
        
        proj_grad_curr = ((loss_curr_1 - loss_curr_2) / (2 * self.args.zo_eps)).item()
        
        # --- B. Compute Gradient at Anchor Model (\tilde{w}) for Active Layers ---
        # Swap weights to Anchor state for active parameters
        
        # Cache current weights
        curr_weights_cache = {}
        for layer_key in active_layers:
            for name, param in self.params_by_layer[layer_key]:
                curr_weights_cache[name] = param.data.clone()
                param.data.copy_(self.anchor_params[name].to(param.device))
            
        # Compute Anchor Grad (using SAME seed and sparse mask)
        self._perturb_active_layers(scaling_factor=1, seed=step_seed)
        loss_anchor_1 = self.zo_forward(model, inputs)
        
        self._perturb_active_layers(scaling_factor=-2, seed=step_seed)
        loss_anchor_2 = self.zo_forward(model, inputs)
        
        self._perturb_active_layers(scaling_factor=1, seed=step_seed) # Restore Anchor
        
        proj_grad_anchor = ((loss_anchor_1 - loss_anchor_2) / (2 * self.args.zo_eps)).item()
        
        # Restore Current Weights
        for layer_key in active_layers:
            for name, param in self.params_by_layer[layer_key]:
                param.data.copy_(curr_weights_cache[name])
            
        # --- C. Update Parameters with IPW & SVRG ---
        lr = self._get_learning_rate()
        
        # Iterate over active layers
        for layer_key in active_layers:
            count = self.current_layer_counts_map[layer_key]
            
            # Deterministic seed per layer (matching _perturb_active_layers logic)
            torch.manual_seed(step_seed + layer_key)
            
            prob = probs_map[layer_key]
            
            # IPW Weight calculation
            raw_ipw = 1.0 / (prob * self.num_active_draws + 1e-8)
            ipw_weight = min(raw_ipw, self.args.adalezo_ipw_clip)
            
            scale_factor = ipw_weight * count
            
            for name, param in self.params_by_layer[layer_key]:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
                
                # SVRG Correction: (g(w) - g(anchor)) + mu
                g_curr = proj_grad_curr * z
                g_anchor = proj_grad_anchor * z
                mu = self.anchor_mu[name].to(param.device)
                
                # Variance Reduced Gradient
                vr_grad = g_curr - g_anchor + mu
                
                # Apply IPW Scaling and Count Multiplier
                final_grad = vr_grad * scale_factor
                
                # Weight Decay
                if self.args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name:
                    final_grad += self.args.weight_decay * param.data
                
                # Update
                param.data -= lr * final_grad
            
        # --- D. Update Bandit Rewards ---
        # We use the magnitude of the gradient difference as a heuristic for "informativeness" of the layer
        # or simply the magnitude of the current projected gradient (similar to AdaLeZO original)
        step_reward = abs(proj_grad_curr) 
        
        for layer_key in active_layers:
            idx = self.sorted_layer_keys.index(layer_key)
            
            count = self.current_layer_counts_map[layer_key]
            self.layer_counts[idx] += count
            
            # EMA Reward Update
            self.layer_avg_rewards[idx] += (step_reward - self.layer_avg_rewards[idx]) / self.layer_counts[idx]
        
        return loss_curr_1

    def _apply_global_noise(self, model, seed, scaling_factor):
        """Apply noise to ALL parameters (for Anchor update)."""
        torch.manual_seed(seed)
        # Using params_by_layer is safer as it covers all groups we care about
        for layer_key, param_list in self.params_by_layer.items():
            for name, param in param_list:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
                param.data += scaling_factor * z * self.args.zo_eps

    def _perturb_active_layers(self, scaling_factor, seed):
        """
        Apply perturbations only to the layers selected by the Bandit.
        Must match the seeding logic used in _ada_svrg_step update phase.
        """
        for layer_key in self.current_active_layers:
            # Seed depends on layer key to be unique but reproducible
            torch.manual_seed(seed + layer_key)
            
            for name, param in self.params_by_layer[layer_key]:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
                param.data += scaling_factor * z * self.args.zo_eps