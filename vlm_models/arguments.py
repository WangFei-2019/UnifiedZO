from dataclasses import dataclass, field
from typing import Optional

@dataclass
class VLMModelArguments:
    """
    Arguments pertaining to which VLM model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained VLM model or model identifier from huggingface.co/models (e.g., llava-hf/llava-1.5-7b-hf)"}
    )
    vision_tower: Optional[str] = field(
        default=None,
        metadata={"help": "Specify the vision tower model name if it differs from the default one in the VLM."}
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the vision tower during ZO optimization. Highly recommended to save memory."}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the LLM backbone."}
    )
    freeze_mm_projector: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the multi-modal projector."}
    )
    quantize_llm: bool = field(
        default=True,
        metadata={"help": "Whether to apply simulated quantization to the LLM backbone."}
    )
    quantize_vision: bool = field(
        default=False,
        metadata={"help": "Whether to apply simulated quantization to the Vision Tower."}
    )
    
    quantized_bit: int = field(
        default=-1,
        metadata={"help": "Target bit-width for quantization (e.g., 2, 4, 8). Default is 4."}
    )


@dataclass
class VLMDataArguments:
    """
    Arguments pertaining to what multimodal data we are going to input our model for training.
    """
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data JSON file (e.g., 'scienceqa', 'mathvista')."}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the folder containing the training images."}
    )