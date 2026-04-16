import torch
import logging
from transformers import AutoProcessor, AutoModelForVision2Seq
from vlm_sim_quant import apply_vlm_simulated_quantization 

logger = logging.getLogger(__name__)

def load_vlm_and_processor(model_args, training_args=None):
    logger.info(f"Loading processor from {model_args.model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    # Left padding is generally safer for batched generation/inference
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    target_dtype = torch.bfloat16 if getattr(training_args, 'bf16', False) else torch.float16
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=target_dtype
    )

    quantize_vision = getattr(model_args, 'quantize_vision', False)
    quantize_llm = getattr(model_args, 'quantize_llm', True)
    
    freeze_vision = getattr(model_args, 'freeze_vision_tower', True)
    freeze_llm = getattr(model_args, 'freeze_llm', False)
    freeze_projector = getattr(model_args, 'freeze_mm_projector', False)

    quant_method = getattr(model_args, 'quant_method', getattr(training_args, 'quant_method', 'none'))
    quant_bits = getattr(model_args, 'quantized_bit', getattr(training_args, 'quantized_bit', -1))

    if quant_method in ["sim_quant", "qzo", "lqzo"] and quant_bits > 0:
        logger.info(f"Triggering dynamic FakeQuant injection for {quant_method.upper()} with {quant_bits}-bit...")

        group_size = getattr(model_args, 'group_size', getattr(training_args, 'group_size', 128))
        model = apply_vlm_simulated_quantization(
            model, 
            bits=quant_bits,
            group_size=group_size,
            quantize_llm=quantize_llm, 
            quantize_vision=quantize_vision
        )
    else:
        logger.info(f"Quantization Skipped. (Method: {quant_method}, Bits: {quant_bits})")

    # 3. Enforce precise gradient requirements for ZO scaling / Finetuning
    logger.info("Enforcing parameter requires_grad rules...")
    for name, param in model.named_parameters():
        # Default policy: freeze everything
        param.requires_grad = False
        
        # [Component A]: Vision Tower
        if "vision_tower" in name:
            if not freeze_vision:
                if quant_bits > 0 and quantize_vision:
                    # QZO/LQZO logic: Only tune scales and bias
                    if "scales" in name or "bias" in name: param.requires_grad = True
                else:
                    param.requires_grad = True
                    
        # [Component B]: LLM Backbone
        elif "language_model" in name or "lm_head" in name:
            if not freeze_llm:
                if quant_bits > 0 and quantize_llm:
                    # QZO/LQZO logic: Tune scales, bias, and preserve high-precision lm_head
                    if "scales" in name or "bias" in name or "lm_head" in name: 
                        param.requires_grad = True
                else:
                    param.requires_grad = True
                    
        # [Component C]: Multi-Modal Projector
        elif "multi_modal_projector" in name:
            if not freeze_projector:
                param.requires_grad = True

    # 4. Verification and Summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Frozen Parameters: {frozen_params:,}")

    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor