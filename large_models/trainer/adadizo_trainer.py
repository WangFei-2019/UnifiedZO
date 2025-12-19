import torch
import copy
import numpy as np
from .adalezo_trainer import AdaLeZOTrainer
from .dizo_trainer import DiZOConstraint  # Assuming dizo_trainer is in the same directory
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaDiZOTrainer(AdaLeZOTrainer):
    """
    Implementation of AdaDiZO (Adaptive Layer-wise DiZO).
    
    Combination of:
    1. AdaLeZO: Adaptive layer selection for sparse gradient estimation (Efficiency).
    2. DiZO: Periodic projected gradient descent steps using learned constraints (Regularization/Stability).
    """

    def __init__(self, model, args, **kwargs):
        # Initialize AdaLeZO (Bandit logic, layer grouping)
        super().__init__(model, args, **kwargs)
        
        # --- DiZO Initialization ---
        self.dizo_interval = getattr(args, "dizo_interval", 100)
        self.dizo_iters = getattr(args, "dizo_iters", 5)
        self.zo_eps_proj = getattr(args, "zo_eps_projection", 1e-3)
        self.step_size_proj = getattr(args, "step_size_projection", 0.1)
        self.clip_range = getattr(args, "clip_range", 1e-4)
        self.norm_mode = getattr(args, "norm_mode", "l2")

        logger.info("Initializing AdaDiZO: Creating anchor model for projection...")
        
        # Create Anchor Model (Fixed reference point)
        # In AdaDiZO, we still need a global anchor to constrain the trajectory
        self.anchor_model = copy.deepcopy(model)
        for param in self.anchor_model.parameters():
            param.requires_grad = False
            
        # Initialize DiZO Constraint Manager
        # We use the same exclude logic as the base trainer (e.g. freezing embeddings)
        self.exclude_list = [n for n, p in model.named_parameters() if not p.requires_grad]
        self.dizo_constraint = DiZOConstraint(model, exclude_list=self.exclude_list, norm_mode=self.norm_mode)
        self.dizo_constraint.to(model.device)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        AdaDiZO Step:
        1. Perform standard AdaLeZO step (Select sparse layers -> ZO Estimate -> IPW Update).
        2. Periodically perform DiZO projection (Optimize Gamma -> Project Weights).
        """
        
        # 1. Standard AdaLeZO Update
        # This handles the sparse perturbation and parameter update
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 2. Periodic DiZO Projection
        # Typically runs less frequently (e.g., every 100 steps) to correct trajectory
        if self.state.global_step > 0 and self.state.global_step % self.dizo_interval == 0:
            self.dizo_step(model, inputs)

        return loss

    def dizo_step(self, model, inputs):
        """
        Executes the DiZO projection logic.
        Note: Even though AdaLeZO updates sparsely, the DiZO projection is applied 
        globally (or to all eligible parameters) to ensure the entire model stays 
        within the trusted region defined by the Anchor.
        """
        logger.info(f"[AdaDiZO] Running projection at step {self.state.global_step}...")
        
        # Ensure modules are on the correct device
        self.dizo_constraint.to(model.device)
        if next(self.anchor_model.parameters()).device != model.device:
            self.anchor_model.to(model.device)

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

        # Sync gamma with current distances on first run or re-init
        if self.dizo_constraint.init:
            for i, (name, gamma) in enumerate(self.dizo_constraint.constraints.named_parameters()):
                gamma.data = ts[i]
            self.dizo_constraint.init = False

        # Optimization loop for Gamma (Constraint Parameters)
        # We perform a few ZO steps to find the optimal projection radius
        for _ in range(self.dizo_iters):
            self._optimize_gamma_one_step(model, inputs, ts)

        # Final Apply: Project model parameters using the optimized gamma
        with torch.no_grad():
            constraint_iterator = iter(self.dizo_constraint.constraints)
            self.dizo_constraint.apply_constraints(model, self.anchor_model, constraint_iterator, save_alpha=False)

    def _optimize_gamma_one_step(self, model, inputs, ts):
        """
        One ZO step to update gamma. 
        Copied/Adapted from DiZOTrainer logic.
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
        
        # Forward (+) - Using BaseZOTrainer's zo_forward
        loss1 = self.zo_forward(model, inputs)
        
        # Reverse constraints
        self.dizo_constraint.reverse_constraints(model, self.anchor_model)
        
        # 2. Perturb Gamma (-)
        self.dizo_constraint.perturb_gamma(-2, ts, zs, tau, zo_eps)
        
        # Apply constraints temporarily
        constraint_iterator = iter(self.dizo_constraint.constraints)
        self.dizo_constraint.apply_constraints(model, self.anchor_model, constraint_iterator, save_alpha=True)
        
        # Forward (-)
        loss2 = self.zo_forward(model, inputs)
        
        # Reverse constraints
        self.dizo_constraint.reverse_constraints(model, self.anchor_model)
        
        # Restore Gamma (to 0 state)
        self.dizo_constraint.perturb_gamma(1, ts, zs, tau, zo_eps)

        # 3. Update Gamma
        grad = (loss1 - loss2) / (2 * zo_eps)
        
        for i, (name, gamma) in enumerate(self.dizo_constraint.constraints.named_parameters()):
            tmp_z = zs[name]
            update = step_size * ts[i] * grad * tmp_z
            
            new_val = gamma.data - update
            lower = (1 - tau) * ts[i]
            upper = (1 + tau) * ts[i]
            gamma.data = torch.clamp(new_val, lower, upper)