import logging
import os
import sys
import json
from dataclasses import asdict
import wandb
import torch
from transformers import (
    AutoConfig, 
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    HfArgumentParser, 
    DefaultDataCollator,
    set_seed
)

# --- Path Setup ---
# Add the parent directory to sys.path to allow importing from 'zo_core'
# structure:
# root/
#   zo_core/
#   vit_models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Core Imports ---
# Importing generic ZO components (Arguments, Trainer Factory) from the core module
from zo_core.arguments import ZOTrainingArguments
from zo_core.trainer import get_trainer_class
from zo_core.utils import result_file_tag, write_metrics_to_file

# --- Local Imports ---
# Importing vision-specific tasks and data processing utilities
from vision_tasks import get_vision_task
from data_utils import process_vision_dataset

# Setup logging configuration
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main execution entry point for Vision Transformer (ViT) ZO training.
    """
    
    # 1. Argument Parsing
    # We reuse the ZOTrainingArguments from the core module as ZO parameters are universal.
    parser = HfArgumentParser((ZOTrainingArguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    # Initialize Weights & Biases (WandB) logging if requested
    if "wandb" in args.report_to and args.local_rank <= 0:
        wandb.init(
            project=f"ZO_ViT_{args.trainer}",
            name=f"{args.task_name}-{args.model_name.split('/')[-1]}-{args.tag if args.tag else ''}",
            config=asdict(args)
        )

    if args.local_rank <= 0:
        logger.info(f"Training Arguments: {args}")

    # Set random seed for reproducibility
    set_seed(args.seed)

    # 2. Setup Vision Task
    # Loads dataset metadata (e.g., number of labels, label mappings)
    task = get_vision_task(args.task_name)
    
    # 3. Load Model and Image Processor
    logger.info(f"Loading Vision Model: {args.model_name}")
    
    # Load Image Processor (handles resizing, normalization config)
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    
    # Load Vision Model
    # Note: We explicitly map label IDs to ensure correct classification head initialization
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=task.num_labels,
        id2label=task.id2label,
        label2id=task.label2id,
        trust_remote_code=True,
        ignore_mismatched_sizes=True # Allow resizing head for fine-tuning on new datasets
    )

    # 4. Inject Parameter Efficient Fine-Tuning (PEFT) - Optional
    # Logic for LoRA/Prefix Tuning can be added here similar to the NLP run.py.
    # For initial migration, we focus on full-parameter or standard ZO optimization.
    if args.lora:
        from peft import get_peft_model, LoraConfig, TaskType
        logger.info("Injecting LoRA for Vision Task...")
        # ViT target modules are usually "query", "value" or "q_proj", "v_proj"
        peft_config = LoraConfig(
            task_type=None, # TaskType.IMAGE_CLASSIFICATION is generic
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=["query", "value", "key", "dense"] # Common for ViT
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 5. Data Processing
    # Convert raw Hugging Face datasets into PyTorch Datasets with pixel_values
    if args.num_train > 0:
        # Subset sampling logic can be added here if needed
        train_source = task.train_dataset.select(range(min(len(task.train_dataset), args.num_train)))
    else:
        train_source = task.train_dataset

    logger.info("Processing Training Dataset...")
    train_dataset = process_vision_dataset(
        args, 
        train_source, 
        image_processor, 
        is_training=True
    )
    
    logger.info("Processing Evaluation Dataset...")
    # Handle num_eval constraint
    if args.num_eval is not None:
        eval_source = task.eval_dataset.select(range(min(len(task.eval_dataset), args.num_eval)))
    else:
        eval_source = task.eval_dataset

    eval_dataset = process_vision_dataset(
        args, 
        eval_source, 
        image_processor, 
        is_training=False
    )

    # 6. Initialize ZO Trainer
    # We use the DefaultDataCollator which handles stacking pixel_values tensors
    collator = DefaultDataCollator()

    # Retrieve the specific ZO Trainer class (MeZO, LoZO, etc.)
    TrainerClass = get_trainer_class(args)
    logger.info(f"Initializing Trainer Class: {TrainerClass.__name__}")

    # Note: Unlike NLP, we do NOT pass a custom 'zo_evaluator' here initially.
    # We rely on the Trainer's standard evaluation loop which uses the model's loss.
    # Accuracy metrics can be computed via compute_metrics if needed.
    trainer = TrainerClass(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=image_processor, # Passing processor as tokenizer for compatibility logging
        data_collator=collator,
    )

    # 7. Training Loop
    if args.trainer != "none":
        logger.info("Starting Training...")
        train_result = trainer.train()
        
        if args.local_rank <= 0:
            logger.info(f"Training finished. Loss: {train_result.training_loss}")
            
        # Save Model
        if args.save_model:
            output_dir = os.path.join(args.output_dir, "final_model")
            logger.info(f"Saving model to {output_dir}...")
            trainer.save_model(output_dir=output_dir)
            image_processor.save_pretrained(output_dir)

    # 8. Evaluation Loop
    if not args.no_eval:
        logger.info("Starting Evaluation...")
        metrics = trainer.evaluate()
        
        logger.info(f"Evaluation Metrics: {metrics}")
        
        if args.local_rank <= 0:
            fname = result_file_tag(args) + "_vision.json"
            output_path = os.path.join(args.output_dir, fname)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_metrics_to_file(metrics, output_path)

    # Final Logging
    if args.report_to == "wandb" and args.local_rank <= 0:
        wandb.finish()

if __name__ == "__main__":
    main()