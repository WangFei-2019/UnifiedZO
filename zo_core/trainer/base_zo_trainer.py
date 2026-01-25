from typing import Optional
import torch
import numpy as np
from transformers import Trainer
from transformers.utils import logging
from metrics import f1
logger = logging.get_logger(__name__)

class BaseZOTrainer(Trainer):
    """
    Base Class for Zeroth-Order Optimization.
    Handles common logic like optimizer creation (dummy), scheduler, and identifying optimization parameters.
    """

    def __init__(self, model, args =None, zo_evaluator=None, raw_dev_samples=None, raw_test_samples=None, **kwargs):
        # if args.trainer != "regular":
        #     args.lr_scheduler_type = "constant"
        super().__init__(model=model, args=args, **kwargs)
        
        # Identify parameters to optimize (requires_grad=True)
        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        
        self.zo_random_seed = None
        self.projected_grad = 0.0

        self.zo_evaluator = zo_evaluator
        self.raw_dev_samples = raw_dev_samples
        self.raw_test_samples = raw_test_samples

    def create_optimizer(self):
        """
        Override to create a dummy optimizer for ZO. 
        But allow standard optimizer for Regular First-Order training.
        """
        if self.args.trainer == "regular":
            return super().create_optimizer()

        # ZO methods update parameters manually in-place.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        return self.optimizer

    def zo_forward(self, model, inputs):
        """
        Execute a forward pass with gradients disabled to compute loss.
        """
        model.eval()
        
        # Handle non-differentiable objectives (e.g. F1) if configured
        if self.args.non_diff:
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            
            if self.args.n_gpu > 1:
                loss = loss.mean()
        
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Handling non-differentiable objectives (e.g., F1 Score).
        """
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id
            )
            output_text = [self.tokenizer.decode(o[inputs["input_ids"].size(1):], skip_special_tokens=True).strip() for o in outputs]
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
            return -torch.tensor(np.mean(f1s), dtype=torch.float32, device=inputs["input_ids"].device)
        
    def generate_random_noise(self, size, device, dtype, noise_type='Gaussian'):
        """
        Generates random noise tensor of specific shape and type.
        """
        if noise_type == 'Gaussian':
            return torch.normal(mean=0, std=1, size=size, device=device, dtype=dtype)
        elif noise_type == 'Rademacher':
            return torch.randint(0, 2, size=size, device=device, dtype=dtype) * 2 - 1
        else:
            raise NotImplementedError(f"Noise type {noise_type} not implemented.")

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Subclasses must implement the specific ZO step logic.
        """
        if self.args.trainer == "regular":
            return super().training_step(model, inputs, num_items_in_batch)
        else:
            raise NotImplementedError("BaseZOTrainer cannot be used directly. Use MeZOTrainer, LoZOTrainer, etc.")
    
    # Override evaluate to handle dual-set evaluation and logging
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and returns metrics.
        Overrides the default HF Trainer evaluate to use our custom ZO Evaluator
        on BOTH Dev and Test sets.
        """
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # If we are not using the custom evaluator setup (e.g. legacy mode), fallback
        if self.zo_evaluator is not None:
            if self.raw_dev_samples:
                logger.info(f"Computing metrics on Dev Set ({len(self.raw_dev_samples)} samples)...")
                dev_metrics = self.zo_evaluator.evaluate(self.raw_dev_samples, [])

                for key, val in dev_metrics.items():
                    metrics[f"eval_{key}"] = val

            if self.raw_test_samples:
                logger.info(f"Computing metrics on Test Set ({len(self.raw_test_samples)} samples)...")
                test_metrics = self.zo_evaluator.evaluate(self.raw_test_samples, [])
                
                for key, val in test_metrics.items():
                    metrics[f"test_{key}"] = val

        log_data = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        log_data["global_step"] = self.state.global_step
        self.log(log_data)
        
        return metrics

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log to inject projected_grad into logs.
        """
        # Log projected_grad if available
        if hasattr(self, "projected_grad") and self.projected_grad is not None:
            val = self.projected_grad
            
            # Convert tensor to float
            if isinstance(val, torch.Tensor):
                if val.numel() > 1:
                    val = val.mean().item()
                else:
                    val = val.item()
                
            logs["projected_grad"] = val
            
        super().log(logs, start_time)