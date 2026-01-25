import torch
import numpy as np
import math
from collections import deque
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from .adalezo_trainer import AdaLeZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaPZOTrainer(AdaLeZOTrainer):
    """
    Implementation of AdaPZO (AdaLeZO + PseuZO).
    Combines Adaptive Layer-wise selection with Momentum/Sliding Window Zeroth-Order Optimization.
    
    Logic:
    1. AdaLeZO: Selects a subset of active layers using Multi-Armed Bandit.
    2. PseuZO: Maintains a sliding window of output differences (o_diff) and computes gradients w.r.t hidden states.
    3. Hybrid Update: 
       - Projects the "true" hidden-state gradient onto the history of output differences.
       - Reconstructs the parameter update only for the layers that were active in the corresponding history steps.
       - Scales updates using Inverse Probability Weighting (IPW) to correct for sparse selection bias.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        
        # --- PseuZO State Initialization ---
        self.sliding_window = deque(maxlen=self.args.sliding_window_length)
        self.coefficients = []
        self.momentum_fb_min = self.args.momentum_fb_min
        self.momentum_fb_max = self.args.momentum_fb_max
        self.grad_last = None
        
        # State to track epoch changes for momentum update
        self.last_epoch_update = -1
        
        # Initialize momentum coefficients
        # Start with max momentum
        self.reset_momentum_fb(self.momentum_fb_max)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        AdaPZO Training Step.
        """
        model.eval()
        
        # 1. Dynamic Momentum Scheduling
        self._update_momentum_coefficient()

        # 2. Get Gradient on Unperturbed model (loss0, o0, grad_last)
        # We need grad_last for the projection step later
        loss0, o0, grad_last = self.pzo_forward(model, inputs, need_grad=True)
        
        # Handle potential tuple return (e.g. from DataParallel)
        self.grad_last = grad_last[0] if isinstance(grad_last, tuple) else grad_last

        # 3. Perturb Active Layers (+) (Using AdaLeZO's logic)
        self.zo_random_seed = np.random.randint(1000000000)
        self._perturb_active_layers(scaling_factor=1)
        
        # 4. Forward on perturbed model (loss1, o1)
        # We don't need grad here, just the loss and output state
        loss1, o1, _ = self.pzo_forward(model, inputs, need_grad=False)
        
        # Restore active layers (Back to 0)
        self._perturb_active_layers(scaling_factor=-1)
        
        # 5. Calculate Output Difference and Update Window
        if o0 is not None and o1 is not None:
            o_diff = o1 - o0
            
            # Store context in sliding window: 
            # Critical: We must store the SPECIFIC active layers and probs for THIS step
            # to reconstruct the sparse gradient direction correctly later.
            current_active = list(self.current_active_layers)
            current_probs = self.current_layer_probs_map.copy()
            current_counts = self.current_layer_counts_map.copy()
            k_draws = self.num_active_draws
            
            self.sliding_window.append((self.zo_random_seed, o_diff, current_active, current_probs, current_counts, k_draws))
        else:
             logger.warning("AdaPZO: Hidden states missing. Skipping window update.")

        # 6. Calculate Reward for Bandit (Proxy: magnitude of change in loss)
        # Since we only did one perturbation (0 -> +1), we use abs(loss1 - loss0)
        step_reward = abs((loss1 - loss0).item())

        # 7. Update Parameters (Hybrid Update: PseuZO projection + AdaLeZO sparse scaling)
        self._update_adapzo()
        
        # 8. Bandit Re-sampling (from AdaLeZO)
        # Update layer statistics and potentially resample active layers
        self._update_bandit_stats(step_reward)
        
        if self.state.global_step % self.args.adalezo_interval == 0:
            self._resample_layers()

        return loss1

    def _update_adapzo(self):
        """
        Update parameters using PZO projection logic combined with AdaLeZO's IPW and sparse updates.
        """
        lr = self._get_learning_rate()
        args = self.args
        
        # --- 1. Calculate PZO Projection Weights (Dot Products) ---
        dot_products = []
        history_items = [] 
        
        if self.grad_last is None:
            return

        # Iterate over sliding window to compute projections <grad_last, o_diff_i>
        for item in self.sliding_window:
            seed, o_diff, active, probs, counts, k_draws = item
            
            # Align sequence lengths (PseuZO logic for varying seq lengths)
            if o_diff.shape[0] != self.grad_last.shape[0]: 
                continue

            seq_o = o_diff.shape[1]
            seq_g = self.grad_last.shape[1]
            min_seq = min(seq_o, seq_g)
            
            if seq_o > min_seq: curr_o = o_diff[:, -min_seq:, :]
            else: curr_o = o_diff
            
            if seq_g > min_seq: curr_g = self.grad_last[:, -min_seq:, :]
            else: curr_g = self.grad_last
                
            # Compute dot product
            dot = torch.sum(curr_o * curr_g, dim=(-3, -2, -1))
            dot_products.append(dot)
            history_items.append(item)
            
        if not dot_products:
            return

        # Get relevant momentum coefficients (tail of the list matching window size)
        current_coeffs = self.coefficients[-len(dot_products):]
        
        # --- 2. Apply Updates ---
        # We iterate through history to apply momentum-aggregated, sparse updates.
        
        with torch.no_grad():
            for (coeff, dot, item) in zip(current_coeffs, dot_products, history_items):
                seed, _, active_layers, probs_map, counts_map, k_draws = item
                
                # Calculate scalar projection value for this history step
                # project_value ~ (grad . o_diff) / eps
                project_value = coeff * dot.item() / args.zo_eps
                
                # Skip tiny updates for efficiency
                if abs(project_value) < 1e-9: continue

                # Reconstruct Sparse Update for this history item
                # Only iterate over layers that were active in THAT step
                for layer_key in active_layers:
                    prob = probs_map[layer_key]
                    count = counts_map[layer_key]

                    # IPW Calculation (AdaLeZO logic)
                    # Weight = 1 / (prob * k_total)
                    raw_ipw = 1.0 / (prob * k_draws + 1e-8)
                    ipw_weight = min(raw_ipw, args.adalezo_ipw_clip)
                    
                    # Apply Count Scaling (because we sum updates if layer selected multiple times)
                    scale_factor = ipw_weight * count
                    
                    # Regenerate noise 'z' using the specific seed for that step + layer key
                    torch.manual_seed(seed + layer_key)
                    
                    # Retrieve parameters for this layer
                    for name, param in self.params_by_layer[layer_key]:
                        z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, args.perturb_type)
                        
                        # Update Term
                        update = project_value * z * scale_factor
                        
                        # Apply update (w/ Weight Decay logic)
                        if args.weight_decay > 0 and ("bias" not in name and "norm" not in name):
                             param.data -= lr * (update + args.weight_decay * param.data)
                        else:
                             param.data -= lr * update

        # Step LR Scheduler
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def _update_bandit_stats(self, step_reward):
        """
        Update Bandit Statistics (EMA rewards) based on the current step's observation.
        """
        # We update stats for the layers active in the *current* step
        for layer_key in self.current_active_layers:
            idx = self.sorted_layer_keys.index(layer_key)
            count = self.current_layer_counts_map[layer_key]
            
            # Update selection count
            self.layer_counts[idx] += count
            
            # Update EMA Reward
            # Note: We use the same reward for all active layers in this step
            self.layer_avg_rewards[idx] = (1 - self.args.adalezo_ema_alpha) * self.layer_avg_rewards[idx] + self.args.adalezo_ema_alpha * step_reward

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------
    def pzo_forward(self, model, inputs, need_grad=False):
        """
        Executes the PZO-specific forward pass to get (loss, state, grad).
        """
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            # Pass is_pzo_step=True to compatible wrappers
            if need_grad:
                 outputs = model(need_grad=True, is_pzo_step=True, **inputs)
                 if isinstance(outputs.loss, tuple):
                     loss_val, state, grad_last = outputs.loss
                 else:
                     # Fallback
                     loss_val = outputs.loss
                     state, grad_last = None, None
            else:
                 outputs = model(need_grad=False, is_pzo_step=True, **inputs)
                 
                 if isinstance(outputs.loss, tuple):
                     loss_val = outputs.loss[0]
                 else:
                     loss_val = outputs.loss
                     
                 grad_last = None
                 
                 # Retrieve state
                 if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                     state = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
                 elif hasattr(outputs, 'logits') and outputs.logits is not None:
                     state = outputs.logits
                 else:
                     state = None

            if self.args.n_gpu > 1 and loss_val.ndim > 0:
                loss_val = loss_val.mean()

        return loss_val, state, grad_last
    

    def _update_momentum_coefficient(self):
        """
        Uses integer epoch for scheduling to match PseuZO paper logic.
        Updates momentum and resets window ONLY when the epoch integer changes.
        """
        epoch = int(self.state.epoch) if self.state.epoch is not None else 0
        
        # Only update if the epoch has changed.
        if epoch == self.last_epoch_update:
            return
        self.last_epoch_update = epoch

        num_train_epochs = self.state.num_train_epochs if self.state.num_train_epochs is not None else 1
        
        def cyclic_hyperbola(t, T, k):
            if k == 0: return self.momentum_fb_max
            cyc = T // k
            if cyc == 0: cyc = 1
            if t >= cyc * k: return self.momentum_fb_min
            t = t % cyc
            return (self.momentum_fb_max / (1 + 10 * t) if t <= 10 else self.momentum_fb_min)
        
        new_momentum = cyclic_hyperbola(epoch, num_train_epochs, 2)
        
        self.reset_momentum_fb(new_momentum)

    def reset_momentum_fb(self, momentum_fb):
        """
        Clears sliding window when momentum resets to max (Restart).
        """
        self.momentum_fb = momentum_fb
        
        # Reset window only on restart (momentum back to max)
        if self.momentum_fb == self.momentum_fb_max:
             self.sliding_window = deque(maxlen=self.args.sliding_window_length)
             # logger.info(f"AdaPZO: Momentum reset to max ({self.momentum_fb}), sliding window cleared.")
        
        self.coefficients = []
        for i in range(self.args.sliding_window_length):
            if i == 0: 
                self.coefficients.append(1.0)
            else:
                self.coefficients = [co * self.momentum_fb for co in self.coefficients]
                self.coefficients.append(1.0)