import torch
import torch.nn as nn
import copy
import numpy as np
from transformers.utils import logging
from .mezo_trainer import MeZOTrainer
from torch.utils.data import DataLoader

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
        # self.init flag is removed as we handle re-initialization explicitly in the trainer

    def _create_constraints(self, model):
        """Creates learnable constraint parameters (gamma) for each model parameter."""
        for name, param in model.named_parameters():
            if name not in self.exclude_list and param.requires_grad:
                self.constraints_name.append(name)
                # Initialize gamma as a learnable scalar, starting at 0 (will be reset in step)
                temp = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
                self.constraints.append(temp)
                self.id_name_map[len(self.constraints_name) - 1] = name

    def _project_ratio(self, new_param, anchor_param, constraint_val):
        """Calculates the projection ratio alpha."""
        diff = new_param.detach() - anchor_param.detach()

        if "l2" in self.norm_mode:
            # L2 Norm: Global scalar norm
            norm = torch.norm(diff)
        else:
            # MARS-like norm or L1: Sum of abs dimensions excluding the first.
            # This handles row-wise projection for matrices (dim 0 = out_features, dim 1... = in_features)
            # If it is a 1D tensor (e.g. bias), dim 1 doesn't exist, so we sum everything.
            dims = tuple(range(1, diff.dim()))
            if len(dims) == 0:
                norm = torch.sum(torch.abs(diff))
            else:
                # keepdim=True is crucial for broadcasting the division later
                norm = torch.sum(torch.abs(diff), dim=dims, keepdim=True)

        # Avoid division by zero
        # constraint_val is scalar, norm might be vector (in MARS mode) or scalar (in L2 mode)
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
                # alpha broadcasts: scalar for L2, vector [Out, 1] for MARS
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
                # ts[i] is the scalar distance of this parameter block
                limit = ((tau / zo_eps) * ts[i]).item()
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
        
        # DiZO Hyperparameters
        self.dizo_interval = args.dizo_interval
        self.dizo_iters = args.dizo_iters
        self.zo_eps_proj = args.zo_eps_projection
        self.step_size_proj = args.step_size_projection
        self.clip_range = args.clip_range
        self.norm_mode = args.norm_mode

        # Create Anchor Model (Pre-trained / Initial state)
        logger.info("Initializing DiZO: Creating anchor model...")
        self.anchor_model = copy.deepcopy(model)
        for param in self.anchor_model.parameters():
            param.requires_grad = False
        
        # Initialize DiZO Constraint Manager
        self.exclude_list = [n for n, p in model.named_parameters() if not p.requires_grad]
        self.dizo_constraint = DiZOConstraint(model, exclude_list=self.exclude_list, norm_mode=self.norm_mode)
        self.dizo_constraint.to(model.device)
        
        # Iterator holder for DiZO inner loop
        self._dizo_train_iterator = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a MeZO update step, followed conditionally by a DiZO projection step.
        """
        # 1. Standard MeZO Step
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 2. DiZO Projection Step (Periodically)
        if self.state.global_step > 0 and self.state.global_step % self.dizo_interval == 0:
            self.dizo_step(model) # Changed: Don't pass inputs, fetch new ones

        return loss

    def _get_next_batch(self):
        """Helper to get the next batch from the training dataloader."""
        if self._dizo_train_iterator is None:
            dataloader = self.get_train_dataloader()
            self._dizo_train_iterator = iter(dataloader)
        
        try:
            inputs = next(self._dizo_train_iterator)
        except StopIteration:
            # Restart iterator if exhausted
            dataloader = self.get_train_dataloader()
            self._dizo_train_iterator = iter(dataloader)
            inputs = next(self._dizo_train_iterator)
            
        return self._prepare_inputs(inputs)

    def dizo_step(self, model):
        """
        Optimizes the constraints (gamma) and projects the model.
        """
        logger.info(f"Running DiZO projection at step {self.state.global_step}...")
        
        # Ensure modules are on the correct device
        self.dizo_constraint.to(model.device)
        if next(self.anchor_model.parameters()).device != model.device:
            self.anchor_model.to(model.device)

        # 1. Initialization: Reset Gamma to current parameter distances (ts)
        # We MUST re-initialize gamma at the start of every DiZO interval.
        ts = []
        with torch.no_grad():
            for (name, para), anchor in zip(model.named_parameters(), self.anchor_model.parameters()):
                if name in self.dizo_constraint.constraints_name:
                    diff = para.data - anchor.data
                    if "l2" in self.norm_mode:
                        norm = torch.norm(diff)
                    else:
                        # Use scalar sum of abs for 'ts' (distance metric for gamma scaling)
                        norm = torch.sum(torch.abs(diff))
                    ts.append(norm)
            
            # Reset gamma values
            for i, param in enumerate(self.dizo_constraint.constraints):
                param.data = ts[i]

        # 2. Optimization Loop for Gamma
        # Optimize gamma for 'dizo_iters' steps using NEW batches
        for _ in range(self.dizo_iters):
            inputs = self._get_next_batch()
            self._optimize_gamma_one_step(model, inputs, ts)

        # 3. Final Apply: Project model parameters permanently
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
        grad = ((loss1 - loss2) / (2 * zo_eps)).item()
        
        # Update rule: gamma = gamma - lr * grad * z * scaling
        for i, (name, gamma) in enumerate(self.dizo_constraint.constraints.named_parameters()):
            tmp_z = zs[name]

            # Projected Gradient Descent on Gamma
            update = step_size * ts[i].item() * grad * tmp_z.item()
            new_val = gamma.data - update
            
            # Clip gamma to be within [1-tau, 1+tau] * original_distance
            lower = (1 - tau) * ts[i]
            upper = (1 + tau) * ts[i]
            gamma.data = torch.clamp(new_val, lower.item(), upper.item())