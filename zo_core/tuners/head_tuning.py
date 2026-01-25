import logging
import torch

logger = logging.getLogger(__name__)

class HeadTuning:
    """
    Head Tuning implementation for UnifiedZO.
    Freezes the backbone and only optimizes the LM Head.
    """
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.apply()

    def apply(self):
        logger.info("Applying Head Tuning...")
        
        # 1. Freeze the entire model first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. Identify and Unfreeze the LM Head
        head_module = self._get_head_module()
        
        if head_module:
            for name, param in head_module.named_parameters():
                param.requires_grad = True
            
            # Handle case where head might be just a weight tensor not in a module (rare in HF)
            if hasattr(head_module, 'weight'):
                head_module.weight.requires_grad = True
            if hasattr(head_module, 'bias') and head_module.bias is not None:
                head_module.bias.requires_grad = True
                
            logger.info(f"Unfrozen LM Head module: {type(head_module)}")
        else:
            raise ValueError(f"Could not automatically detect LM Head for model: {self.args.model_name}")

        # 3. Log Trainable Parameters
        self._log_trainable_params()

    def _get_head_module(self):
        """
        Helper to find the LM head module based on architecture.
        """
        model = self.model
        
        # Priority 1: Direct attribute access (Standard HF CausalLM)
        if hasattr(model, "lm_head"):
            return model.lm_head
        
        # Priority 2: Architecture specific checks
        if "opt" in self.args.model_name.lower():
            return model.lm_head if hasattr(model, "lm_head") else None
        
        if "gpt2" in self.args.model_name.lower():
            return model.lm_head
            
        # Priority 3: Search by name convention
        for name, module in model.named_modules():
            if name.endswith("lm_head") or name.endswith("embed_out") or name.endswith("output_layer"):
                return module
                
        return None

    def _log_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.6f}"
        )