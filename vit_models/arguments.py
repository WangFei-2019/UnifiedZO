from dataclasses import dataclass, field
from typing import Optional
from zo_core.arguments import ZOTrainingArguments

@dataclass
class ViTZOTrainingArguments(ZOTrainingArguments):
    """
    Inherits from ZOTrainingArguments (zo_core) and adds Vision-specific quantization args.
    """
    quant_method: str = field(default="none", metadata={"help": "Quantization method: 'gptq', 'aqlm', or 'none'."})

    quantized_bit: int = field(
        default=4,
        metadata={"help": "Target bit-width for quantization (e.g., 2, 4, 8). Default is 4."}
    )
    quant_group_size: int = field(
        default=128,
        metadata={"help": "Group size for quantization (e.g., 128, 64). Default is 128."}
    )
