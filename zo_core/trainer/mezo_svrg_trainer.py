import torch
import copy
import numpy as np
from .mezo_trainer import MeZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MeZOSVRGTrainer(MeZOTrainer):
    """
    Implementation of MeZO-SVRG (Stochastic Variance-Reduced Gradient).
    """
    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        
        # SVRG Hyperparameters
        self.svrg_q = args.svrg_q    # Update frequency for anchor (inner loop length)
        self.svrg_k = args.svrg_k    # Number of ZO samples to estimate full gradient (mu)
        
        # Storage for Anchor State
        # We use a dictionary to store CPU tensors to save GPU memory
        self.anchor_params = {} 
        self.anchor_mu = {}     # The "full batch" gradient at the anchor point
        
        # Initialize anchor at step 0
        self.has_initialized_anchor = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        MeZO-SVRG Training Step
        """
        model.eval()

        # 1. Periodic Anchor Update (Outer Loop)
        # Initialize if first step OR if we hit the interval q
        if not self.has_initialized_anchor or (self.state.global_step > 0 and self.state.global_step % self.svrg_q == 0):
            self._update_anchor(model, inputs)
            self.has_initialized_anchor = True

        # 2. Standard MeZO-SVRG Step (Inner Loop)
        loss = self._svrg_inner_step(model, inputs)

        return loss

    def _update_anchor(self, model, inputs):
        """
        Updates the anchor model (\tilde{w}) and calculates the 'full' gradient (\mu).
        """
        logger.info(f"[MeZO-SVRG] Updating anchor at step {self.state.global_step}...")
        
        # 1. Save current parameters as Anchor
        self.anchor_params = {}
        for name, param in self.named_parameters_to_optim:
            self.anchor_params[name] = param.detach().cpu().clone()
        
        # 2. Estimate "Full" Gradient (Mu) at Anchor
        # We average over svrg_k samples to get a lower variance estimate of the anchor gradient
        self.anchor_mu = {name: torch.zeros_like(param) for name, param in self.named_parameters_to_optim}
        
        # Determine scaling factor (accumulate average)
        scale = 1.0 / self.svrg_k
        
        for _ in range(self.svrg_k):
            # Sample z
            seed = np.random.randint(1000000000)
            
            # Forward + / -
            self._perturb_model(model, seed, scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)
            
            self._perturb_model(model, seed, scaling_factor=-2) # +1 -> -1
            loss2 = self.zo_forward(model, inputs)
            
            self._perturb_model(model, seed, scaling_factor=1) # Restore to 0
            
            # Grad estimate
            projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
            
            # Accumulate into Mu
            torch.manual_seed(seed)
            for name, param in self.named_parameters_to_optim:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
                self.anchor_mu[name] += (projected_grad * z * scale).detach()
        
        logger.info(f"[MeZO-SVRG] Anchor updated.")

    def _svrg_inner_step(self, model, inputs):
        """
        Performs the variance-reduced update:
        grad = grad(w_t) - grad(\tilde{w}) + \mu
        """
        # 1. Sample Seed for this step (Shared between w_t and \tilde{w})
        step_seed = np.random.randint(1000000000)
        
        # --- A. Compute Gradient at Current Model (w_t) ---
        self._perturb_model(model, step_seed, scaling_factor=1)
        loss_curr_1 = self.zo_forward(model, inputs)
        
        self._perturb_model(model, step_seed, scaling_factor=-2)
        loss_curr_2 = self.zo_forward(model, inputs)
        
        self._perturb_model(model, step_seed, scaling_factor=1) # Restore
        
        proj_grad_curr = ((loss_curr_1 - loss_curr_2) / (2 * self.args.zo_eps)).item()
        
        # --- B. Compute Gradient at Anchor Model (\tilde{w}) ---
        # We must temporarily load anchor weights to compute grad(\tilde{w}) with SAME noise z
        # Optimization: To avoid full CPU-GPU swap, we can compute perturbations on the fly if memory allows,
        # but swapping is safer for VRAM. For PEFT, this is fast. For Full FT, this is slow.
        
        # 1. Swap w_t -> \tilde{w}
        current_params_cache = {}
        for name, param in self.named_parameters_to_optim:
            current_params_cache[name] = param.data.clone() # Keep on GPU if possible, or move to CPU
            param.data.copy_(self.anchor_params[name].to(param.device))
            
        # 2. Compute Grad(\tilde{w})
        self._perturb_model(model, step_seed, scaling_factor=1)
        loss_anchor_1 = self.zo_forward(model, inputs)
        
        self._perturb_model(model, step_seed, scaling_factor=-2)
        loss_anchor_2 = self.zo_forward(model, inputs)
        
        self._perturb_model(model, step_seed, scaling_factor=1) # Restore anchor
        
        proj_grad_anchor = ((loss_anchor_1 - loss_anchor_2) / (2 * self.args.zo_eps)).item()
        
        # 3. Swap \tilde{w} -> w_t (Restore current model)
        for name, param in self.named_parameters_to_optim:
            param.data.copy_(current_params_cache[name])
        
        # --- C. Update Parameters ---
        # Update rule: w_{t+1} = w_t - lr * ( (grad_curr - grad_anchor) + mu )
        
        lr = self._get_learning_rate()
        torch.manual_seed(step_seed)
        
        for name, param in self.named_parameters_to_optim:
            # Re-generate z
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
            
            # SVRG Gradient Estimate
            # term1: \hat{\nabla} f(w_t) = proj_grad_curr * z
            # term2: \hat{\nabla} f(\tilde{w}) = proj_grad_anchor * z
            # term3: \mu = self.anchor_mu[name]
            
            variance_reduced_grad = (proj_grad_curr - proj_grad_anchor) * z + self.anchor_mu[name].to(param.device)
            
            # Weight decay handling
            if self.args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name:
                variance_reduced_grad += self.args.weight_decay * param.data
            
            # Apply Update
            param.data -= lr * variance_reduced_grad

        return loss_curr_1

    def _perturb_model(self, model, seed, scaling_factor):
        """Helper to apply perturbations given a seed."""
        torch.manual_seed(seed)
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
            param.data += scaling_factor * z * self.args.zo_eps