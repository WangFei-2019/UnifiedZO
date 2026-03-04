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
        metadata={"help": "Whether to freeze the vision tower during Zeroth-Order optimization. Highly recommended to save memory."}
    )
    freeze_mm_projector: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the multi-modal projector."}
    )

@dataclass
class VLMDataArguments:
    """
    Arguments pertaining to what multimodal data we are going to input our model for training.
    """
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data JSON file."}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the folder containing the training images."}
    )