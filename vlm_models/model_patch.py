import torch
from transformers import AutoProcessor
from gptqmodel import GPTQModel

def load_vlm_and_processor(model_args):
    """
    Loads the Vision-Language Model using GPTQ.
    This guarantees compatibility with QZOTrainer/LQZOTrainer which specifically 
    look for 'module.scales' in TritonV2QuantLinear or MarlinQuantLinear.
    """
    print(f"Loading processor from {model_args.model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    # Note: model_args.model_name_or_path MUST point to a GPTQ quantized model!
    # e.g., "TheBloke/llava-v1.5-7b-GPTQ" or similar.
    print(f"Loading GPTQ VLM model {model_args.model_name_or_path}...")
    
    # 使用 GPTQModel 加载，确保底层 Linear 层被替换为 TritonV2QuantLinear
    model = GPTQModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto"
    )

    # 强制将模型标志位设为量化状态，适配 ZO 优化器的逻辑
    model.is_quantized = True

    # Freeze Vision Tower parameters
    if model_args.freeze_vision_tower and hasattr(model, "vision_tower"):
        print("Freezing the vision tower parameters...")
        for param in model.vision_tower.parameters():
            param.requires_grad = False
            
    # Freeze Multi-Modal Projector if specified
    if model_args.freeze_mm_projector and hasattr(model, "multi_modal_projector"):
        print("Freezing the multi-modal projector parameters...")
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

    return model, processor