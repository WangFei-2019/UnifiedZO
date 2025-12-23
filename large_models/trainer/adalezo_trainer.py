import torch
import numpy as np
import math
import re
import os
import json
from collections import Counter
from .base_zo_trainer import BaseZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaLeZOTrainer(BaseZOTrainer):
    """
    Implementation of AdaLeZO (Adaptive Layer-wise Zeroth-Order Optimization) with REPLACEMENT sampling.
    
    Mechanism:
    1. Divides parameters into groups (layers).
    2. Uses Multi-Armed Bandit logic (EMA rewards) to select a sparse subset of layers.
       - Sampling is WITH REPLACEMENT (a layer can be chosen multiple times).
    3. Estimates gradients using ZO only on active layers.
    4. Updates parameters using Inverse Probability Weighting (IPW), scaled by selection counts.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        
        # Initialize AdaLeZO state (Layer grouping and Bandit statistics)
        self.init_adalezo_state(model)
        
        # Track current active layers for the step
        self.current_active_layers = []
        self.current_layer_probs_map = {}
        self.current_layer_counts_map = {}
        self.num_active_draws = 0
        
        # Initial sampling of layers
        self._resample_layers()

    def init_adalezo_state(self, model):
        """
        Parses model parameters to group them by layers and initializes Bandit statistics.
        """
        self.params_by_layer = {}
        
        for name, param in model.named_parameters():
            if not param.requires_grad: 
                continue
            
            # Regex to extract layer index. Supports standard HF models (Llama, OPT, BERT, etc.)
            # Matches patterns like 'layers.0.', 'blocks.1.', 'h.2.'
            match = re.search(r"\.(layers|block|h|blocks)\.(\d+)\.", name)
            
            # Key determination:
            # - Layers: index from regex
            # - Embeddings: -1
            # - Head/Norms: 9999 (Generic high index)
            key = int(match.group(2)) if match else (-1 if "embed" in name else 9999)
            
            if key not in self.params_by_layer: 
                self.params_by_layer[key] = []
            self.params_by_layer[key].append((name, param))
            
        self.sorted_layer_keys = sorted(self.params_by_layer.keys())
        self.num_layers = len(self.sorted_layer_keys)
        
        logger.info(f"[AdaLeZO] Initialized. Total Optimization Groups (Layers): {self.num_layers}")

        # --- Bandit Statistics ---
        # N_i: Selection counts (kept for logging/analysis)
        self.layer_counts = torch.zeros(self.num_layers, device=self.args.device)
        # Q_i: Average reward (gradient magnitude)
        self.layer_avg_rewards = torch.zeros(self.num_layers, device=self.args.device)
        
        # Initialize Probabilities
        if self.args.adalezo_warm_start:
            # Warm start: Bias towards deeper layers (last 40%)
            self.layer_scores = torch.zeros(self.num_layers, device=self.args.device)
            with torch.no_grad():
                for i, key in enumerate(self.sorted_layer_keys):
                    if key > self.num_layers * 0.6:
                        self.layer_scores[i] = 1.0
            self.layer_probs = torch.nn.functional.softmax(self.layer_scores / self.args.adalezo_tau, dim=0)
        else:
            # Uniform initialization
            self.layer_probs = torch.ones(self.num_layers, device=self.args.device) / self.num_layers

        # Adaptive Scaling State (RMSProp-like)
        if self.args.adalezo_layer_momentum:
            self.layer_sq_grads = torch.zeros(self.num_layers, device=self.args.device) 

    def _resample_layers(self):
        """
        Selects active layers using WITH REPLACEMENT sampling.
        """
        # Number of samples to draw (k)
        k = max(1, int(self.args.adalezo_k_ratio * self.num_layers))
        self.num_active_draws = k
        
        # Normalize rewards for numerical stability
        max_reward = self.layer_avg_rewards.max()
        # Use rewards directly as scores (Exploitation)
        scores = self.layer_avg_rewards / max_reward if max_reward > 1e-6 else self.layer_avg_rewards
        
        # Convert scores to probabilities via Softmax (Temperature controlled)
        self.layer_probs = torch.nn.functional.softmax(scores / self.args.adalezo_tau, dim=0)

        # Mix with uniform distribution to ensure exploration (Gamma Mixing)
        # Critical for preventing starvation of low-probability layers
        self.layer_probs = (1 - self.args.adalezo_gamma) * self.layer_probs + self.args.adalezo_gamma * (1.0 / self.num_layers)

        # Sample k indices WITH REPLACEMENT
        active_indices = torch.multinomial(self.layer_probs, k, replacement=True)
        
        # Map indices to layer keys
        active_keys_raw = [self.sorted_layer_keys[i] for i in active_indices.tolist()]
        
        # Count occurrences of each layer (How many times each was selected)
        self.current_layer_counts_map = Counter(active_keys_raw)
        
        # Active layers are the UNIQUE keys selected
        self.current_active_layers = sorted(self.current_layer_counts_map.keys())

        # Store Probabilities for IPW calculation later (only for unique active layers)
        self.current_layer_probs_map = {
            key: self.layer_probs[self.sorted_layer_keys.index(key)].item() 
            for key in self.current_active_layers
        }

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        AdaLeZO Training Step:
        1. Perturb ONLY active layers (+).
        2. Forward.
        3. Perturb ONLY active layers (-).
        4. Forward.
        5. Update parameters.
        """
        model.eval()

        # 1. Sample Seed
        self.zo_random_seed = np.random.randint(1000000000)
        
        # 2. Perturb Active Layers (+)
        # Note: Even if a layer was selected 5 times, we only perturb it ONCE here.
        # The '5 times' logic is handled in the update step (gradient scaling).
        self._perturb_active_layers(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        # 3. Perturb Active Layers (-) : Move from +1 to -1 state (subtract 2)
        self._perturb_active_layers(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        # 4. Calculate Projected Gradient Estimate (Scalar proxy)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        # Restore Active Layers to original state (add 1)
        self._perturb_active_layers(scaling_factor=1)
        
        # 5. Update Parameters & Bandit Stats
        self._update_adalezo()
        
        # 6. Periodic Re-sampling (Stickiness)
        if self.state.global_step % self.args.adalezo_interval == 0:
            self._resample_layers()
            # Optional: Log probs
            self._log_probs()

        return loss1

    def _perturb_active_layers(self, scaling_factor):
        """
        Apply perturbations only to the layers selected by the Bandit.
        """
        # Deterministic seeding per layer to ensure consistency
        base_seed = self.zo_random_seed
        
        for layer_key in self.current_active_layers:
            # Seed depends on layer key to be unique but reproducible
            torch.manual_seed(base_seed + layer_key)
            
            for name, param in self.params_by_layer[layer_key]:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                param.data += scaling_factor * z * self.args.zo_eps

    def _update_adalezo(self):
        """
        Update parameters using IPW, scaled by selection counts.
        """
        args = self.args
        lr = self._get_learning_rate()
        
        step_reward = abs(self.projected_grad)

        # Iterate only over UNIQUE active layers
        for layer_key in self.current_active_layers:
            prob = self.current_layer_probs_map[layer_key]
            count = self.current_layer_counts_map[layer_key] # Number of times selected
            
            # --- IPW Calculation (With Replacement) ---
            # Unbiased Estimator Formula: 
            # Update = Sum_{j=1 to k} [ I(layer selected at j) * (g / (k * p)) ]
            #        = count * (g / (k * p))
            
            # Denominator uses Total Draws (k), not unique count
            raw_ipw = 1.0 / (prob * self.num_active_draws + 1e-8)
            ipw_weight = min(raw_ipw, args.adalezo_ipw_clip)
            
            # [MODIFIED] Scale factor includes the count (multiplicity)
            scale_factor = ipw_weight * count

            # --- Optional: Layer-wise Momentum (Adaptive Scaling) ---
            if args.adalezo_layer_momentum:
                idx = self.sorted_layer_keys.index(layer_key)
                
                # Estimate variance. 
                # Note: We update momentum stats based on "one observation" of the gradient
                layer_grad_est = self.projected_grad * ipw_weight
                current_energy = layer_grad_est ** 2
                
                self.layer_sq_grads[idx] = args.adalezo_beta * self.layer_sq_grads[idx] + (1 - args.adalezo_beta) * current_energy
                denom = torch.sqrt(self.layer_sq_grads[idx]) + 1e-8
                
                scale_factor = scale_factor / denom

            # --- Parameter Update ---
            torch.manual_seed(self.zo_random_seed + layer_key)
            for name, param in self.params_by_layer[layer_key]:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                
                # Update term = g * z * IPW * Count
                update_term = self.projected_grad * z * scale_factor
                
                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                         param.data.add_(update_term + args.weight_decay * param.data, alpha=-lr)
                    else:
                         param.data.add_(update_term, alpha=-lr)
                else:
                    param.data.add_(update_term, alpha=-lr)

            # --- Update Bandit Stats ---
            # We update the stats ONCE per step per layer, regardless of 'count'.
            # Reason: The gradient signal 'step_reward' is observed only once this step.
            idx = self.sorted_layer_keys.index(layer_key)
            self.layer_counts[idx] += count # Track total allocated budget
            
            # EMA Update
            self.layer_avg_rewards[idx] = (1 - args.adalezo_ema_alpha) * self.layer_avg_rewards[idx] + args.adalezo_ema_alpha * step_reward

    def _log_probs(self):
        """
        Helper to log probability evolution to file.
        """
        if self.layer_probs is None: return
        
        data = {
            "step": self.state.global_step,
            "probs": self.layer_probs.detach().cpu().tolist(),
            "active": self.current_active_layers,
            "counts": dict(self.current_layer_counts_map), # Log counts for debug
            "normalized_rewards": self.layer_avg_rewards.detach().cpu().tolist(), 
        }
        
        # Save to output dir
        save_path = os.path.join(self.args.output_dir, "adalezo_probs.jsonl")
        with open(save_path, "a") as f:
            f.write(json.dumps(data) + "\n")