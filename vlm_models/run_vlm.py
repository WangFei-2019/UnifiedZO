import sys
import logging
import transformers
from transformers import set_seed
import wandb
from dataclasses import asdict
from arguments import VLMModelArguments, VLMDataArguments
from data_utils import DataCollatorForVLM, ScienceQADataset, MathVistaDataset
from model_patch import load_vlm_and_processor

sys.path.append('..') 
from zo_core.trainer.lqzo_trainer import LQZOTrainer
from zo_core.trainer.qzo_trainer import QZOTrainer
from zo_core.arguments import ZOTrainingArguments

# Import the ZO forward wrapper critical for variance reduction
from zo_core.trainer.utils import forward_wrap_with_option_len

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    parser = transformers.HfArgumentParser((VLMModelArguments, VLMDataArguments, ZOTrainingArguments))
    
    # Now, `training_args` is an instance of ZOTrainingArguments, containing BOTH
    # standard HF arguments and your specific ZO arguments (like zo_eps, lozo_rank).
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if hasattr(training_args, 'seed'):
        set_seed(training_args.seed)
        logger.info(f"Global seed set to {training_args.seed} for reproducible ZO perturbations.")

    # 1. Load Model and Processor (Pass training_args for bf16 dtype alignment)
    model, processor = load_vlm_and_processor(model_args, training_args)

    if "wandb" in training_args.report_to:
        model_short_name = model_args.model_name_or_path.split('/')[-1]
        dataset_name = data_args.data_path if data_args.data_path else "unknown_dataset"
        zo_method = training_args.trainer.upper()
        
        full_config = {}
        full_config.update(asdict(model_args))
        full_config.update(asdict(data_args))
        full_config.update(training_args.to_dict())

        wandb.init(
            project=f"UnifiedZO_VLM_{dataset_name}",
            name=f"{zo_method}-{model_short_name}-LR{training_args.learning_rate}-EPS{training_args.zo_eps}",
            config=full_config,
            tags=[zo_method, dataset_name, "VLM"]
        )
        logger.info("WandB explicitly initialized with strict hyperparameter tracking.")

    if getattr(training_args, 'only_train_option', False):
        if not hasattr(model, 'original_forward'):
            model.original_forward = model.forward
        
        # Monkey-patch the forward pass dynamically based on ZO trainer type
        model.forward = forward_wrap_with_option_len.__get__(model, type(model))
        logger.info("Successfully injected `forward_wrap_with_option_len`. ZO Loss will be strictly evaluated on target options.")
    else:
        logger.warning("WARNING: `only_train_option` is False. The ZO gradient estimator will suffer from high variance on VLM tasks!")

    # 2. Prepare ScienceQA Dataset
    logger.info("Preparing ScienceQA dataset...")
    if data_args.data_path == "mathvista":
        train_dataset = MathVistaDataset(split="train", processor=processor)
        eval_dataset = MathVistaDataset(split="testmini", processor=processor)
    else:
        train_dataset = ScienceQADataset(split="train", processor=processor)

    eval_dataset = ScienceQADataset(split="validation", processor=processor)

    # 3. Initialize the appropriate ZO Trainer
    data_collator = DataCollatorForVLM(processor=processor)
    
    trainer_cls = LQZOTrainer if getattr(training_args, 'zo_optim', '') == "lqzo" else QZOTrainer
    logger.info(f"Initializing {trainer_cls.__name__} for VLM fine-tuning...")
    
    trainer = trainer_cls(
        model=model,
        args=training_args,  # Pass the consolidated arguments
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Evaluation dataset activated
        data_collator=data_collator,
    )

    # 4. Execute Training
    logger.info("Starting Zeroth-Order Optimization on VLM...")
    trainer.train()

    # Clean up patching post-training
    if hasattr(model, 'original_forward'):
        model.forward = model.original_forward
        del model.original_forward

    if "wandb" in training_args.report_to:
        wandb.finish()

if __name__ == "__main__":
    main()