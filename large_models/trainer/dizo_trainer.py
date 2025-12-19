import torch
import torch.nn as nn
import copy
import numpy as np
from transformers.utils import logging
from .mezo_trainer import MeZOTrainer

logger = logging.get_logger(__name__)

class DiZOConstraint(nn.Module):
    """
    Helper class for DiZO to manage constraints and projections.
    Adapted from the original DiZO implementation to fit UnifiedZO.
    """
    def __init__(self, model, exclude_list=[], norm_mode='l2'):
        super().__init__()
        self.norm_mode = norm_mode
        self.exclude_list = exclude_list
        self.constraints = nn.ParameterList([])
        self.constraints_name = []
        self.id_name_map = {}
        self.alpha = {}
        
        # Initialize constraints (gamma) for trainable parameters
        self._create_constraints(model)
        self.init = True

    def _create_constraints(self, model):
        """Creates learnable constraint parameters (gamma) for each model parameter."""
        for name, param in model.named_parameters():
            if name not in self.exclude_list and param.requires_grad:
                self.constraints_name.append(name)
                # Initialize gamma as a learnable scalar, starting at 0
                temp = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
                self.constraints.append(temp)
                self.id_name_map[len(self.constraints_name) - 1] = name

    def _project_ratio(self, new_param, anchor_param, constraint_val):
        """Calculates the projection ratio alpha."""
        diff = new_param.detach() - anchor_param.detach()
        
        if "l2" in self.norm_mode:
            norm = torch.norm(diff)
        else:
            # MARS-like norm (sum of abs dimensions excluding first) or L1
            norm = torch.sum(torch.abs(diff)) 

        # Avoid division by zero
        ratio = constraint_val / (norm + 1e-8)
        return ratio

    def apply_constraints(self, model, anchor_model, constraint_iterator, save_alpha=False):
        """
        Projects the model parameters based on the current constraints (gamma).
        new_theta = anchor + alpha * (new_theta - anchor)
        """
        for (name, new_para), anchor_para in zip(model.named_parameters(), anchor_model.parameters()):
            if name in self.constraints_name:
                constraint = next(constraint_iterator)
                
                # Calculate projection ratio
                alpha = self._project_ratio(new_para, anchor_para, constraint)
                
                if save_alpha:
                    self.alpha[name] = alpha
                
                # Apply projection
                # v = direction * alpha
                v = (new_para.detach() - anchor_para.detach()) * alpha
                temp = v + anchor_para.detach()
                
                # Update model parameter in-place
                new_para.data.copy_(temp)

    def reverse_constraints(self, model, anchor_model):
        """
        Reverses the projection to restore the model state before calculating the gradient for gamma.
        This allows estimating gradients w.r.t gamma without permanently modifying theta during the estimation step.
        """
        for (name, new_para), anchor_para in zip(model.named_parameters(), anchor_model.parameters()):
            if name in self.constraints_name:
                alpha = self.alpha.get(name, 1.0)
                # v = (current - anchor) / alpha -> restores original direction magnitude
                v = (new_para.detach() - anchor_para.detach()) / alpha
                temp = v + anchor_para.detach()
                new_para.data.copy_(temp)

    def perturb_gamma(self, scaling_factor, ts, zs, tau, zo_eps):
        """
        Perturbs the constraint parameters (gamma) for ZO gradient estimation.
        gamma' = gamma + scaling_factor * z * eps
        """
        for i, (name, gamma) in enumerate(self.constraints.named_parameters()):
            # If zs is empty, generate new noise (first perturbation)
            if name not in zs:
                z = torch.normal(0, 1, size=(1,), device=gamma.device, dtype=gamma.dtype)
                # Clip noise based on tau and ts (current parameter distance)
                limit = (tau / zo_eps) * ts[i]
                z = torch.clamp(z, -limit, limit)
                zs[name] = z
            else:
                z = zs[name]

            gamma.data = gamma.data + scaling_factor * z * zo_eps
        return zs

class DiZOTrainer(MeZOTrainer):
    """
    DiZO Trainer Implementation for UnifiedZO.
    Inherits from MeZOTrainer to use MeZO as the base optimizer, 
    and adds the DiZO projection step periodically.
    """

    def __init__(self, model, args=None, **kwargs):
        super().__init__(model, args=args, **kwargs)
        
        # DiZO Hyperparameters (Extract from args or set defaults)
        self.dizo_interval = getattr(args, "dizo_interval", 100) # How often to run DiZO
        self.dizo_iters = getattr(args, "dizo_iters", 5)         # How many ZO steps for gamma per interval
        self.zo_eps_proj = getattr(args, "zo_eps_projection", 1e-3)
        self.step_size_proj = getattr(args, "step_size_projection", 0.1)
        self.clip_range = getattr(args, "clip_range", 1e-4) # tau
        self.norm_mode = getattr(args, "norm_mode", "l2")

        # Create Anchor Model (Pre-trained / Initial state)
        # We move it to CPU to save memory unless necessary, or keep on device if small enough
        logger.info("Initializing DiZO: Creating anchor model...")
        self.anchor_model = copy.deepcopy(model)
        for param in self.anchor_model.parameters():
            param.requires_grad = False
        # If GPU memory is tight, you might want to move anchor to CPU and move back when needed
        # self.anchor_model.to('cpu') 

        # Initialize DiZO Constraint Manager
        # Exclude list typically includes embeddings or unrelated layers
        self.exclude_list = [n for n, p in model.named_parameters() if not p.requires_grad]
        self.dizo_constraint = DiZOConstraint(model, exclude_list=self.exclude_list, norm_mode=self.norm_mode)
        self.dizo_constraint.to(model.device)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a MeZO update step, followed conditionally by a DiZO projection step.
        """
        # 1. Standard MeZO Step (updates model parameters)
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 2. DiZO Projection Step (Periodically)
        if self.state.global_step > 0 and self.state.global_step % self.dizo_interval == 0:
            self.dizo_step(model, inputs)

        return loss

    def dizo_step(self, model, inputs):
        """
        Optimizes the constraints (gamma) and projects the model.
        """
        logger.info(f"Running DiZO projection at step {self.state.global_step}...")
        
        # Ensure modules are on the correct device
        self.dizo_constraint.to(model.device)
        # Ensure anchor is on the correct device (if it was offloaded)
        if next(self.anchor_model.parameters()).device != model.device:
            self.anchor_model.to(model.device)

        # DiZO Optimization Loop for Gamma
        # We optimize gamma for 'dizo_iters' steps using the CURRENT batch 'inputs'
        # (In original DiZO, it iterates over a loader, here we simplify to use current batch 
        # or we could fetch more batches if dataloader access was easy)
        
        # Calculate current distances (ts) for initialization
        ts = []
        with torch.no_grad():
            for (name, para), anchor in zip(model.named_parameters(), self.anchor_model.parameters()):
                if name in self.dizo_constraint.constraints_name:
                    if "l2" in self.norm_mode:
                        norm = torch.norm(para.data - anchor.data)
                    else:
                        norm = torch.sum(torch.abs(para.data - anchor.data))
                    ts.append(norm)

        # If it's the first time or re-init is needed, sync gamma with current distances
        if self.dizo_constraint.init:
            for i, (name, gamma) in enumerate(self.dizo_constraint.constraints.named_parameters()):
                gamma.data = ts[i]
            self.dizo_constraint.init = False # Keep init False to maintain learned history if desired, or reset

        # Optimization of Gamma
        for _ in range(self.dizo_iters):
            self._optimize_gamma_one_step(model, inputs, ts)

        # Final Apply: Project model parameters permanently using the optimized gamma
        with torch.no_grad():
            constraint_iterator = iter(self.dizo_constraint.constraints)
            self.dizo_constraint.apply_constraints(model, self.anchor_model, constraint_iterator, save_alpha=False)
        
        logger.info("DiZO projection applied.")

    def _optimize_gamma_one_step(self, model, inputs, ts):
        """
        One ZO step to update gamma (constraint parameters).
        """
        zs = {}
        tau = self.clip_range
        zo_eps = self.zo_eps_proj
        step_size = self.step_size_proj
        
        # 1. Perturb Gamma (+)
        self.dizo_constraint.perturb_gamma(1, ts, zs, tau, zo_eps)
        
        # Apply constraints temporarily
        constraint_iterator = iter(self.dizo_constraint.constraints)
        self.dizo_constraint.apply_constraints(model, self.anchor_model, constraint_iterator, save_alpha=True)
        
        # Forward (+)
        loss1 = self.zo_forward(model, inputs)
        
        # Reverse constraints (restore model to pre-projection state)
        self.dizo_constraint.reverse_constraints(model, self.anchor_model)
        
        # 2. Perturb Gamma (-) (From +1 to -1 requires -2 perturbation)
        self.dizo_constraint.perturb_gamma(-2, ts, zs, tau, zo_eps)
        
        # Apply constraints temporarily
        constraint_iterator = iter(self.dizo_constraint.constraints)
        self.dizo_constraint.apply_constraints(model, self.anchor_model, constraint_iterator, save_alpha=True)
        
        # Forward (-)
        loss2 = self.zo_forward(model, inputs)
        
        # Reverse constraints
        self.dizo_constraint.reverse_constraints(model, self.anchor_model)
        
        # Restore Gamma to original state (From -1 to 0 requires +1)
        self.dizo_constraint.perturb_gamma(1, ts, zs, tau, zo_eps)

        # 3. Update Gamma
        # Grad estimate: (L1 - L2) / (2 * eps)
        grad = (loss1 - loss2) / (2 * zo_eps)
        
        # Update rule: gamma = gamma - lr * grad * z * scaling
        # Note: We scale update by ts[i] as in original DiZO paper/code
        for i, (name, gamma) in enumerate(self.dizo_constraint.constraints.named_parameters()):
            tmp_z = zs[name]
            # Projected Gradient Descent on Gamma
            update = step_size * ts[i] * grad * tmp_z
            new_val = gamma.data - update
            
            # Clip gamma to be within [1-tau, 1+tau] * original_distance
            # This prevents the constraint from deviating too wildly from the current trajectory
            lower = (1 - tau) * ts[i]
            upper = (1 + tau) * ts[i]
            gamma.data = torch.clamp(new_val, lower, upper)