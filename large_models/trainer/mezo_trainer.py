import torch
import numpy as np
from .base_zo_trainer import BaseZOTrainer

class MeZOTrainer(BaseZOTrainer):
    """
    Implementation of MeZO (Memory-efficient Zeroth-Order Optimization).
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        MeZO Step:
        1. Sample z
        2. L1 = Loss(theta + z)
        3. L2 = Loss(theta - z)
        4. Grad ~ (L1 - L2) / (2 eps) * z
        5. Update theta
        """
        model.eval()

        # 1. Sample Seed
        self.zo_random_seed = np.random.randint(1000000000)
        
        # 2. Forward +
        self._perturb_mezo(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        # 3. Forward - (From +1 to -1 requires -2 step)
        self._perturb_mezo(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        # 4. Calculate Projected Gradient Estimate
        self.projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps)
        
        # Restore parameters to original state (from -1 to 0 requires +1)
        self._perturb_mezo(scaling_factor=1)
        
        # 5. Update Parameters
        self._update_mezo()
        
        return loss1

    def _perturb_mezo(self, scaling_factor):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
            param.data += scaling_factor * z * self.args.zo_eps

    def _update_mezo(self):
        torch.manual_seed(self.zo_random_seed)
        lr = self._get_learning_rate()
        
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
            
            # Update rule: theta = theta - lr * (grad_est + weight_decay)
            update = self.projected_grad * z
            
            # Apply weight decay
            if self.args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                update += self.args.weight_decay * param.data
                
            param.data -= lr * update