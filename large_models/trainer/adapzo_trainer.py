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
        
        # Initialize momentum coefficients
        self.reset_momentum_fb(self.momentum_fb_max)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        AdaPZO Training Step.
        """
        model.eval()
        
        # 1. Dynamic Momentum Scheduling (from PseuZO)
        self._update_momentum_coefficient()

        # 2. Get Gradient on Unperturbed model (loss0, o0, grad_last)
        # Using PZO custom forward to get gradient w.r.t hidden states/logits
        loss0, o0, grad_last = self.pzo_forward(model, inputs, need_grad=True)
        # Store gradient for the update phase
        self.grad_last = grad_last[0] if isinstance(grad_last, tuple) else grad_last

        # 3. Perturb Active Layers (+) (from AdaLeZO)
        # Generate a fresh seed for this step
        self.zo_random_seed = np.random.randint(1000000000)
        
        # Apply perturbation only to active layers
        self._perturb_active_layers(scaling_factor=1)
        
        # 4. Forward on perturbed model (loss1, o1)
        loss1, o1, _ = self.pzo_forward(model, inputs, need_grad=False)
        
        # Restore active layers (Back to 0)
        self._perturb_active_layers(scaling_factor=-1)
        
        # 5. Calculate Output Difference
        o_diff = o1 - o0
        
        # Store context in sliding window: 
        # Must store counts and K (num_active_draws) for correct IPW reconstruction later.
        current_active = list(self.current_active_layers)
        current_probs = self.current_layer_probs_map.copy()
        current_counts = self.current_layer_counts_map.copy() # Store counts
        k_draws = self.num_active_draws # Store K
        
        self.sliding_window.append((self.zo_random_seed, o_diff, current_active, current_probs, current_counts, k_draws))
        
        # 6. Calculate Reward for Bandit (Proxy: abs(loss1 - loss0))
        # This measures the sensitivity of the selected layers for the bandit algorithm.
        step_reward = abs((loss1 - loss0).item())

        # 7. Update Parameters (Hybrid Update)
        self._update_adapzo(step_reward)
        
        # 8. Bandit Re-sampling (from AdaLeZO)
        if self.state.global_step % self.args.adalezo_interval == 0:
            self._resample_layers()

        return loss1

    def _update_adapzo(self, step_reward):
        """
        Update parameters using PZO projection logic combined with AdaLeZO's IPW and sparse updates.
        """
        lr = self._get_learning_rate()
        args = self.args
        
        # --- 1. Calculate PZO Projection Weights (Dot Products) ---
        dot_products = []
        history_items = [] 
        
        for item in self.sliding_window:
            seed, o_diff, active, probs, counts, k_draws = item
            
            # Align sequence lengths if necessary (truncation logic from PZO)
            if o_diff.shape[0] != self.grad_last.shape[0]: continue

            seq_o = o_diff.shape[1]
            seq_g = self.grad_last.shape[1]
            min_seq = min(seq_o, seq_g)
            
            if seq_o > min_seq: curr_o = o_diff[:, -min_seq:, :]
            else: curr_o = o_diff
            
            if seq_g > min_seq: curr_g = self.grad_last[:, -min_seq:, :]
            else: curr_g = self.grad_last
                
            # Compute dot product: <o_diff, grad_last>
            dot = torch.sum(curr_o * curr_g, dim=(-3, -2, -1))
            dot_products.append(dot)
            history_items.append((seed, active, probs, counts, k_draws))
            
        if not dot_products:
            return

        # Get relevant momentum coefficients
        current_coeffs = self.coefficients[-len(dot_products):]
        
        # --- 2. Apply Updates ---
        # We iterate through history to apply momentum-aggregated updates.
        # Each history item represents a direction 'z' (sparse) that was taken in the past.
        
        for (coeff, dot, (seed, active_layers, probs_map, counts_map, k_draws)) in zip(current_coeffs, dot_products, history_items):
            # Calculate scalar projection value
            project_value = coeff * dot.item() / args.zo_eps
            
            # Reconstruct Sparse Update for this history item
            for layer_key in active_layers:
                prob = probs_map[layer_key]
                
                # Retrieve count for this specific history step
                count = counts_map[layer_key]

                # IPW Calculation
                raw_ipw = 1.0 / (prob * k_draws + 1e-8)
                ipw_weight = min(raw_ipw, args.adalezo_ipw_clip)
                
                # Apply Count Scaling
                scale_factor = ipw_weight * count
                
                # Regenerate noise 'z' using the stored seed + layer_key
                torch.manual_seed(seed + layer_key)
                
                for name, param in self.params_by_layer[layer_key]:
                    z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, args.perturb_type)
                    
                    # Update Term: projection * z * IPW * Count
                    update = project_value * z * scale_factor
                    
                    # Apply update (with optional weight decay)
                    if args.weight_decay > 0 and "bias" not in name and "norm" not in name:
                         param.data -= lr * (update + args.weight_decay * param.data)
                    else:
                         param.data -= lr * update

        # --- 3. Update Bandit Statistics (for CURRENT active layers) ---
        # We update the bandit stats based on the *current* step's sensitivity (step_reward).
        for layer_key in self.current_active_layers:
            idx = self.sorted_layer_keys.index(layer_key)
            # Increment by current count
            count = self.current_layer_counts_map[layer_key]
            self.layer_counts[idx] += count
            self.layer_avg_rewards[idx] += (step_reward - self.layer_avg_rewards[idx]) / self.layer_counts[idx]

    # --------------------------------------------------------------------------
    # Utilities ported from PZOTrainer
    # --------------------------------------------------------------------------
    def pzo_forward(self, model, inputs, need_grad=False):
        """
        Executes the PZO-specific forward pass.
        Passes 'is_pzo_step=True' to ensure wrappers return the necessary states/logits.
        """
        model.eval()
        inputs = self._prepare_inputs(inputs)
        
        # [Correction]: Pass is_pzo_step=True to tell wrapper we are training, not evaluating.
        # This ensures we get the state back even if need_grad=False.
        outputs = model(need_grad=need_grad, is_pzo_step=True, **inputs)
        
        # Unpack based on return type
        if isinstance(outputs.loss, tuple):
            # [Step 1]: Wrapper returned (loss, state, grad)
            loss_val, state, grad_last = outputs.loss
        else:
            # [Step 2]: Wrapper returned Object
            loss_val = outputs.loss
            grad_last = None
            
            # Extract state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # PZO (hidden version): state is in hidden_states[-1]
                state = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
            elif hasattr(outputs, 'logits') and outputs.logits is not None:
                 # PZO (logits version): state is logits
                state = outputs.logits
            else:
                state = None 

        return loss_val, state, grad_last
    

    def _update_momentum_coefficient(self):
        """
        Updates the momentum decay factor based on a cyclic schedule.
        """
        epoch = self.state.epoch
        num_train_epochs = self.state.num_train_epochs if self.state.num_train_epochs is not None else 1
        
        def cyclic_hyperbola(t, T, k):
            cyc = T // k
            if t >= cyc * k: return self.momentum_fb_min
            t = t % cyc
            return (self.momentum_fb_max / (1 + 10 * t) if t <= 10 else self.momentum_fb_min)
        
        new_momentum = cyclic_hyperbola(epoch, num_train_epochs, 2)
        self.reset_momentum_fb(new_momentum)

    def reset_momentum_fb(self, momentum_fb):
        """
        Resets and pre-calculates momentum coefficients.
        """
        self.momentum_fb = momentum_fb
        self.coefficients = []
        for i in range(self.args.sliding_window_length):
            if i == 0: 
                self.coefficients.append(1.0)
            else:
                self.coefficients = [co * self.momentum_fb for co in self.coefficients]
                self.coefficients.append(1.0)