import torch
import numpy as np
import math
import logging
from collections import deque
from .base_zo_trainer import BaseZOTrainer

logger = logging.getLogger(__name__)

class PZOTrainer(BaseZOTrainer):
    """
    Implementation of PseuZO (Momentum/Sliding Window ZO).
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        # PseuZO state (Sliding Window & Momentum)
        self.sliding_window = deque(maxlen=self.args.sliding_window_length)
        self.coefficients = []
        self.momentum_fb_min = 0.0
        self.momentum_fb_max = 1.0 
        self.grad_last = None
        
        # State to track epoch changes for momentum update (Fix: Added state tracking)
        self.last_epoch_update = -1
        
        # Init coefficients list
        # We start with max momentum
        self.reset_momentum_fb(self.momentum_fb_max)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        PseuZO Step:
        1. Calculate gradient w.r.t hidden states (grad_last) on unperturbed model.
        2. Perturb model.
        3. Calculate change in output states (o = o1 - o0).
        4. Store (seed, o) in sliding window.
        """
        model.eval()
        
        # --- 1. Dynamic Momentum Scheduling ---
        self._update_momentum_coefficient()

        # --- 2. Get Gradient on Unperturbed model (and state o0) ---
        # need_grad=True to get grad_last
        loss0, o0, grad_last = self.pzo_forward(model, inputs, need_grad=True)
        
        # Handling potential tuple return from DataParallel or specific models
        # Ensure grad_last is a single tensor corresponding to the batch
        self.grad_last = grad_last[0] if isinstance(grad_last, tuple) else grad_last 

        # --- 3. Perturb ---
        seed = np.random.randint(1e9)
        self.zo_random_seed = seed # Save for update
        self._perturb_mezo(1) 
        
        # --- 4. Forward on perturbed model (only need state o1, no grad needed) ---
        loss1, o1, _ = self.pzo_forward(model, inputs, need_grad=False)
        
        # Restore model
        self._perturb_mezo(-1) 
        
        # --- 5. Calculate Difference in Output Space ---
        if o0 is not None and o1 is not None:
            o_diff = o1 - o0
            # Store in window
            self.sliding_window.append((seed, o_diff))
        else:
            # Fallback if states are missing (should not happen in correct PseuZO setup)
            logger.warning("PseuZO: Hidden states not returned, skipping sliding window update.")
        
        # --- 6. Update ---
        self._update_pzo()

        return loss1

    def _update_pzo(self):
        """
        PseuZO Update:
        Project the current gradient (grad_last) onto the history of output differences (o_diff).
        """
        lr = self._get_learning_rate()
        
        dot_products = []
        seeds = []
        
        if self.grad_last is None:
            return

        # Calculate dot products <grad_last, o_diff>
        for seed, o_diff in self.sliding_window:
            # Handle potential sequence length mismatch (truncation)
            # o_diff and grad_last should be (Batch, Seq, Hidden)
            
            if o_diff.shape[0] != self.grad_last.shape[0]:
                continue # Batch size mismatch

            seq_o = o_diff.shape[1]
            seq_g = self.grad_last.shape[1]
            min_seq = min(seq_o, seq_g)
            
            if seq_o > min_seq:
                curr_o = o_diff[:, -min_seq:, :]
                curr_g = self.grad_last
            elif seq_g > min_seq:
                curr_o = o_diff
                curr_g = self.grad_last[:, -min_seq:, :]
            else:
                curr_o = o_diff
                curr_g = self.grad_last
                
            # Dot product over (Batch, Seq, Dim) -> Result is scalar per batch item? 
            # Original implementation sums over all dims including batch: dim=(-3, -2, -1)
            # assuming shape is (Batch, Seq, Dim)
            dot = torch.sum(curr_o * curr_g, dim=(-3, -2, -1))
            dot_products.append(dot)
            seeds.append(seed)
            
        # Apply Coefficients (Momentum)
        # Using the last N coefficients corresponding to the items in window
        if len(dot_products) > 0:
            # Get corresponding coefficients. 
            # self.coefficients has length equal to max window size, usually.
            # We take the tail of coefficients matching the number of valid history items.
            current_coeffs = self.coefficients[-len(dot_products):]
            
            final_weights = [co * dot.item() / self.args.zo_eps for co, dot in zip(current_coeffs, dot_products)]
            
            # Aggregate Parameter Updates
            # Note: This loop can be slow. In optimized implementations, we might aggregate seeds first.
            with torch.no_grad():
                for i, (project_value, seed) in enumerate(zip(final_weights, seeds)):
                    if abs(project_value) < 1e-8: continue 

                    torch.manual_seed(seed)
                    for name, param in self.named_parameters_to_optim:
                        z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)

                        # Update rule
                        if self.args.weight_decay > 0 and ("bias" not in name and "norm" not in name):
                            param.data -= lr * (project_value * z + self.args.weight_decay * param.data)
                        else:
                            param.data -= lr * (project_value * z)

        # Step LR Scheduler
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def pzo_forward(self, model, inputs, need_grad=False):
        """
        PZO forward wrapper.
        """
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            
            # Ensure the model wrapper (e.g. unifiedzo model wrapper) supports 'is_pzo_step'
            # and returns (loss, hidden_states, gradient) when need_grad is True.
            if need_grad:
                 outputs = model(need_grad=True, is_pzo_step=True, **inputs)
                 # Expecting outputs.loss to be a tuple: (actual_loss, hidden_state, gradient)
                 if isinstance(outputs.loss, tuple):
                     loss_val, state, grad_last = outputs.loss
                 else:
                     # Fallback/Error handling if wrapper isn't set up correctly
                     loss_val = outputs.loss
                     state = None 
                     grad_last = None
                     logger.warning("PZO: Model did not return tuple (loss, state, grad). Check model wrapper.")
            else:
                 outputs = model(need_grad=False, is_pzo_step=True, **inputs)
                 
                 if isinstance(outputs.loss, tuple):
                     loss_val = outputs.loss[0]
                 else:
                     loss_val = outputs.loss
                     
                 grad_last = None
                 
                 # Try to retrieve hidden states for o1
                 if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                     state = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
                 elif hasattr(outputs, 'logits') and outputs.logits is not None:
                     state = outputs.logits
                 else:
                     state = None

            if self.args.n_gpu > 1 and loss_val.ndim > 0:
                loss_val = loss_val.mean()

        return loss_val, state, grad_last

    def _perturb_mezo(self, scaling_factor):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
            param.data += scaling_factor * z * self.args.zo_eps

    # --- Momentum Scheduling Logic (FIXED) ---

    def _update_momentum_coefficient(self):
        # Use integer epoch to match original behavior (step-wise schedule per epoch)
        # self.state.epoch is float (e.g., 0.1, 0.2). Cast to int.
        epoch = int(self.state.epoch) if self.state.epoch is not None else 0
    
        if epoch == self.last_epoch_update:
            return
        self.last_epoch_update = epoch

        num_train_epochs = self.state.num_train_epochs if self.state.num_train_epochs is not None else 1
        
        def cyclic_hyperbola(t, T, k):
            if k == 0: return self.momentum_fb_max # Avoid div by zero
            cyc = T // k
            if cyc == 0: cyc = 1
            
            if t >= cyc * k:
                return self.momentum_fb_min
            t = t % cyc
            # Original formula: max / (1 + 10*t) if t <= 10 else min
            return (self.momentum_fb_max / (1 + 10 * t) if t <= 10 else self.momentum_fb_min)

        # Update momentum_fb based on schedule
        # Using 2 cycles as per original default
        new_momentum = cyclic_hyperbola(epoch, num_train_epochs, 2)
        
        # Only reset if value changes (or at least check logic inside reset)
        # But crucially, we must check if we are 'restarting' to clear the window.
        # In original code, reset_momentum_fb handles the logic of clearing window 
        # when momentum hits max (which happens at start of cycle).
        self.reset_momentum_fb(new_momentum)

    def reset_momentum_fb(self, momentum_fb):
        old_momentum = getattr(self, 'momentum_fb', None)
        self.momentum_fb = momentum_fb
        
        # Clear sliding window if momentum resets to max (Cycle Restart)
        # Original code: if self.momentum_fb == self.momentum_fb_max: ... deque(maxlen=...)
        # Because we now only call this once per epoch, this correctly clears 
        # the window only at the start of a new cycle (when momentum jumps back to max).
        if self.momentum_fb == self.momentum_fb_max:
             self.sliding_window = deque(maxlen=self.args.sliding_window_length)
             # logger.info(f"PZO: Momentum reset to max ({self.momentum_fb}), sliding window cleared.")

        self.coefficients = []
        # Re-calculate coefficients list based on new momentum
        for i in range(self.args.sliding_window_length):
            if i == 0:
                self.coefficients.append(1.0)
            else:
                self.coefficients = [co * self.momentum_fb for co in self.coefficients]
                self.coefficients.append(1.0)