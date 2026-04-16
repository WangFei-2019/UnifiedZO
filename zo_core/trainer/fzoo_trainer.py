import torch
import numpy as np
from .base_zo_trainer import BaseZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class FZooTrainer(BaseZOTrainer):
    """
    Implementation of FZOO (as defined in the provided mezotrainer.py).
    Key features:
    1. Uses multiple perturbations (N) per step.
    2. Estimates gradient using variance reduction (dividing by std of losses).
    3. Maintains FZOO-specific state (N, seeds, random values).
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        # FZOO Specific State
        self.rand_values = None
        self.zo_random_seeds = None
        # Map fzoo_n argument to N
        self.N = args.fzoo_n 
        self.losses = None # Placeholder if needed for extended logic
        self.thr_step = args.fzoo_thre
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        FZOO Training Step logic:
        1. zo_step: Perturb N times, collect losses, compute normalized projected gradient.
        2. zo_update: Apply updates using the pre-computed projected gradient.
        """
        model.eval()

        # --- Step 1: Estimate Gradient (zo_step) ---
        # This calculates self.projected_grad and loss_tensor
        loss_tensor = self.zo_step(model, inputs)

        # --- Step 2: Update Parameters (zo_update) ---
        self.zo_update(model)
        
        return loss_tensor

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb parameters. 
        Note: FZOO reference uses Rademacher noise (binary +-1).
        """
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        # Iterate over params. FZOO reference code generates a random value 'rand_value' 
        # alongside the loop, though it appears unused in the arithmetic of the reference.
        # We preserve the logic to match random state consumption.
        for (name, param), rand_value in zip(self.named_parameters_to_optim, self.rand_values):
            # Rademacher noise: (0 or 1) * 2 - 1 -> -1 or 1
            z = torch.randint(0, 2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype) * 2 - 1
            param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_step(self, model, inputs):
        """
        Executes N perturbations to estimate the gradient.
        Returns:
            loss_tensor: The loss of the unperturbed model.
        Side Effect:
            Sets self.projected_grad (The estimated gradient magnitude vector).
            Sets self.zo_random_seeds (Seeds used for the N perturbations).
        """
        args = self.args

        # 1. Initialize random values state for the step
        # Note: These values are generated but logic inside perturb loop suggests they are auxiliary
        self.rand_values = [torch.rand(1).item() for _ in range(len(self.named_parameters_to_optim))]
        
        # 2. Reset Seeds container
        self.zo_random_seeds = []
        loss1s = []
        
        # Determine perturbation times (N or N/2 logic from reference)
        # Reference: "perturbTimes = self.N // 2 ... if self.losses is None: perturbTimes = self.N"
        perturbTimes = self.N if self.losses is None else (self.N // 2)

        with torch.no_grad():
            for i in range(perturbTimes):
                # Generate a unique seed for this perturbation instance
                self.zo_random_seed = np.random.randint(1000000000)
                self.zo_random_seeds.append(self.zo_random_seed)
                
                # FZOO logic: +1 perturbation
                self.zo_perturb_parameters(scaling_factor=1)
                loss1 = self.zo_forward(model, inputs).detach()
                loss1s.append(loss1)
                
                # FZOO logic: Restore (Reference code does scaling_factor=-1 here to restore? 
                # Note: If we added +1, adding -1 brings us back to 0. 
                # The reference says: "self.zo_perturb_parameters(scaling_factor=-1)" inside the loop.
                # Since we need to start fresh for the next iteration (or calculate difference),
                # we must restore.
                self.zo_perturb_parameters(scaling_factor=-1)

            # 3. Calculate Stats
            loss1s = torch.tensor(loss1s, dtype=torch.float32)
            # Calculate standard deviation of the losses from perturbations
            std = torch.std(loss1s, unbiased=False)
            
            # 4. Get Baseline Loss (Unperturbed)
            loss_tensor_gpu = self.zo_forward(model, inputs).detach()
            loss_tensor_cpu = loss_tensor_gpu.cpu()

        # 5. Compute Projected Gradient
        # Formula: (Loss_perturbed - Loss_baseline) / (N * std)
        # Note: This creates a vector of gradients corresponding to each seed index
        if std == 0 or torch.isnan(std):
            # Fallback to avoid division by zero if all losses are identical
            std = 1.0
             
        self.projected_grad = ((loss1s - loss_tensor_cpu) / (perturbTimes * std)).item()
        # self.projected_grad = (loss1s - loss_tensor_cpu) / (perturbTimes * self.args.zo_eps) # Unlike the original LOZO code, which is biased, here we change to remove the bias. 

        # Logging as per reference
        if self.state.global_step % 10 == 0:
            logger.info(f"Step {self.state.global_step} - std: {std:.4e}, loss: {loss_tensor_cpu:.4f}")

        return loss_tensor_gpu

    def zo_update(self, model):
        """
        Apply the estimated gradients to parameters.
        """
        args = self.args
        lr = self._get_learning_rate()
        
        # Iterate through the seeds used in the step
        for idx, zo_random_seed in enumerate(self.zo_random_seeds):
            # Reset the random seed to regenerate the exact same 'z' noise mask
            torch.manual_seed(zo_random_seed)     
            
            for (name, param), rand_value in zip(self.named_parameters_to_optim, self.rand_values):
                # Re-generate Rademacher noise
                z = torch.randint(0, 2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype) * 2 - 1
                
                # Update Rule: theta = theta - lr * (grad_estimate_i * z)
                # Note: grad_estimate_i is self.projected_grad[idx]
                param.data = param.data - lr * (self.projected_grad[idx] * z)

        # Step standard scheduler if it exists (usually BaseZOTrainer handles LR manually, 
        # but this call updates internal step counters if needed)
        if self.lr_scheduler:
            self.lr_scheduler.step()