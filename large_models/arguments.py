from dataclasses import dataclass, field
from zo_core.arguments import ZOTrainingArguments

@dataclass
class LMZOTrainingArguments(ZOTrainingArguments):
    """
    Arguments for Large Language Models fine-tuning with ZO.
    """