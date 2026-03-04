import torch
import logging
from transformers import AutoProcessor
from gptqmodel import GPTQModel

logger = logging.getLogger(__name__)

def load_vlm_and_processor(model_args, training_args=None):
    """
    Robust instantiation of Vision-Language Models for Zeroth-Order Optimization.
    Ensures precise DType alignment and strict padding token initialization.
    """
    logger.info(f"Loading processor from {model_args.model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True, 
        # use_fast=False
        )

    if not hasattr(processor, "tokenizer") or not hasattr(processor, "image_processor"):
        logger.error(f"Catastrophic failure: Expected a multi-modal Processor, but got {type(processor)}.")
        raise ValueError(
            "Your local model directory is heavily missing multi-modal configuration files "
            "(e.g., 'preprocessor_config.json' or 'processor_config.json'). "
            "AutoProcessor degraded to a pure text tokenizer. Please download the missing files."
        )

    if processor.tokenizer.pad_token is None:
        logger.warning("Tokenizer lacks a pad_token. Assigning eos_token as pad_token to prevent collator crashes.")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
    # Ensure padding side is left for decoder-only generation (standard practice)
    processor.tokenizer.padding_side = "left"

    # Determine optimal precision for unquantized modules (Vision Tower, Projector)
    target_dtype = torch.float16
    if training_args is not None and getattr(training_args, 'bf16', False):
        target_dtype = torch.bfloat16

    logger.info(f"Loading GPTQ VLM model {model_args.model_name_or_path} with precision {target_dtype}...")
    
    # Load model with explicit torch_dtype to prevent FP32 memory explosion
    model = GPTQModel.from_quantized(
        model_args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=target_dtype
    )

    # Ensure the library successfully wrapped the linear layers into Triton/Marlin QuantLinear
    # This is an absolute prerequisite for our QZOTrainer to find the `.scales` parameters.
    quantized_layer_found = any(
        "QuantLinear" in str(type(module)) for module in model.modules()
    )
    if not quantized_layer_found:
        logger.error("VLM Architecture mismatch: GPTQModel failed to inject QuantLinear layers. QZOTrainer will fail.")
        raise RuntimeError("Model loading failed: No QuantLinear layers detected. Check if gptqmodel supports this specific LLaVA variant.")
    
    model.is_quantized = True

    if getattr(model_args, 'freeze_vision_tower', True):
        logger.info("Enforcing strict gradient freeze on the Vision Tower...")
        # Supports LLaVA standard names, add Qwen/other architectures dynamically
        vision_modules = ["vision_tower", "visual", "vision_model"]
        for module_name in vision_modules:
            if hasattr(model, module_name):
                for param in getattr(model, module_name).parameters():
                    param.requires_grad = False
                logger.info(f"Successfully frozen: {module_name}")
                break
            
    if getattr(model_args, 'freeze_mm_projector', False):
        logger.info("Enforcing strict gradient freeze on the Multi-Modal Projector...")
        projector_modules = ["multi_modal_projector", "projector"]
        for module_name in projector_modules:
            if hasattr(model, module_name):
                for param in getattr(model, module_name).parameters():
                    param.requires_grad = False
                logger.info(f"Successfully frozen: {module_name}")
                break

    # For ZO Optimization, we optionally sync pad_token_id to model config
    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor