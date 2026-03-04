import torch
import logging
from transformers import AutoProcessor, AutoModelForVision2Seq
from vlm_sim_quant import apply_vlm_simulated_quantization 

logger = logging.getLogger(__name__)

def load_vlm_and_processor(model_args, training_args=None):
    logger.info(f"Loading processor from {model_args.model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
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

    # --- 步骤 1：读取控制参数 ---
    # 假设你在 argument 中新增了这几个参数，如果没写，默认为标准的只微调 LLM 模式
    quantize_vision = getattr(model_args, 'quantize_vision', False)
    quantize_llm = getattr(model_args, 'quantize_llm', True)
    
    freeze_vision = getattr(model_args, 'freeze_vision_tower', True)
    freeze_llm = getattr(model_args, 'freeze_llm', False)
    freeze_projector = getattr(model_args, 'freeze_mm_projector', False)

    quant_method = getattr(model_args, 'quant_method', 'sim_quant')
    
    quant_bits = getattr(model_args, 'quantized_bit', getattr(training_args, 'quantized_bit', 4))
    
    if quant_method in ["sim_quant", "qzo", "lqzo"]:
        logger.info(f"Triggering dynamic FakeQuant injection with {quant_bits}-bit...")
        model = apply_vlm_simulated_quantization(
            model, 
            bits=quant_bits,
            quantize_llm=quantize_llm, 
            quantize_vision=quantize_vision
        )
        model.is_quantized = True
    else:
        model.is_quantized = False

    # --- 步骤 3：执行梯度/微调控制 ---
    logger.info("Enforcing parameter requires_grad rules based on configuration...")
    for name, param in model.named_parameters():
        # 默认全部冻结
        param.requires_grad = False
        
        # [控制区块 A]：Vision Tower
        if "vision_tower" in name:
            if not freeze_vision:
                # 如果量化了 Vision，那么只微调 scales (和 bias)
                if quantize_vision:
                    if "scales" in name or "bias" in name: param.requires_grad = True
                # 如果没量化，说明是全参微调 (FP16/BF16)
                else:
                    param.requires_grad = True
                    
        # [控制区块 B]：LLM Backbone
        elif "language_model" in name or "lm_head" in name:
            if not freeze_llm:
                if quantize_llm:
                    # 注意：lm_head 通常保持高精度全参微调
                    if "scales" in name or "bias" in name or "lm_head" in name: 
                        param.requires_grad = True
                else:
                    param.requires_grad = True
                    
        # [控制区块 C]：多模态投影层 Projector
        elif "multi_modal_projector" in name:
            if not freeze_projector:
                param.requires_grad = True

    # 打印可训练参数供核对
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {trainable_params:,}")

    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor