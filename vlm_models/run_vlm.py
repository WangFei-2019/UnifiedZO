import transformers
from arguments import VLMModelArguments, VLMDataArguments
from data_utils import DataCollatorForVLM, ScienceQADataset
from model_patch import load_vlm_and_processor

import sys
sys.path.append('..') 
from zo_core.trainer.lqzo_trainer import LQZOTrainer
from zo_core.trainer.qzo_trainer import QZOTrainer
from zo_core.arguments import ZOTrainingArguments

def main():
    parser = transformers.HfArgumentParser((VLMModelArguments, VLMDataArguments, ZOTrainingArguments, transformers.TrainingArguments))
    model_args, data_args, zo_args, training_args = parser.parse_args_into_dataclasses()

    # 1. Load Model and Processor
    model, processor = load_vlm_and_processor(model_args)

    # 2. Prepare ScienceQA Dataset
    # We instantiate the custom dataset, automatically downloading and filtering the image subset.
    print("Preparing training dataset...")
    train_dataset = ScienceQADataset(split="train", processor=processor)
    
    # Optional: Prepare validation dataset for evaluation during training
    # eval_dataset = ScienceQADataset(split="validation", processor=processor)

    # 3. Initialize the appropriate ZO Trainer based on arguments
    data_collator = DataCollatorForVLM(processor=processor)
    
    trainer_cls = LQZOTrainer if zo_args.zo_optim == "lqzo" else QZOTrainer
    print(f"Initializing {trainer_cls.__name__} for VLM fine-tuning...")
    
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # Uncomment if you want evaluation steps
        data_collator=data_collator,
        zo_args=zo_args
    )

    # 4. Execute Training
    print("Starting Zeroth-Order Optimization on VLM...")
    trainer.train()

if __name__ == "__main__":
    main()