import sys
import os
import logging
import transformers
from transformers import set_seed
import wandb
import torch
import numpy as np
from dataclasses import asdict
from arguments import VLMModelArguments, VLMDataArguments
from data_utils import DataCollatorForVLM, ScienceQADataset, MathVistaDataset
from model_patch import load_vlm_and_processor

sys.path.append('..') 
from zo_core.trainer import get_trainer_class
from zo_core.arguments import ZOTrainingArguments
from zo_core.trainer.utils import forward_wrap_with_option_len

from evaluation import preprocess_logits_for_metrics, compute_token_metrics, evaluate_vlm_predictions

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = transformers.HfArgumentParser((VLMModelArguments, VLMDataArguments, ZOTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if getattr(training_args, "remove_unused_columns", True):
        logger.info("Architecture Enforcer: Auto-disabling `remove_unused_columns` for VLM.")
        training_args.remove_unused_columns = False

    if hasattr(training_args, 'seed'):
        set_seed(training_args.seed)

    model, processor = load_vlm_and_processor(model_args, training_args)

    if "wandb" in training_args.report_to:
        wandb.init(
            project=f"UnifiedZO_VLM_{data_args.data_path}",
            name=f"{training_args.trainer.upper()}-{model_args.model_name_or_path.split('/')[-1]}",
            config={**asdict(model_args), **asdict(data_args), **training_args.to_dict()}
        )

    if getattr(training_args, 'only_train_option', False):
        if not hasattr(model, 'original_forward'):
            model.original_forward = model.forward
        model.forward = forward_wrap_with_option_len.__get__(model, type(model))
        logger.info("Successfully injected `forward_wrap_with_option_len`.")

    if data_args.data_path == "mathvista":
        logger.info("Initializing MathVista dataset...")
        train_dataset = MathVistaDataset(split="train", processor=processor)
        eval_dataset = MathVistaDataset(split="testmini", processor=processor)
    else:
        logger.info("Initializing ScienceQA dataset...")
        train_dataset = ScienceQADataset(split="train", processor=processor)
        eval_dataset = ScienceQADataset(split="validation", processor=processor)

    data_collator = DataCollatorForVLM(processor=processor)
    TrainerClass = get_trainer_class(training_args) 
    
    logger.info(f"Instantiating {TrainerClass.__name__} for VLM optimization...")
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_token_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if training_args.trainer.lower() != "none":
        logger.info("Starting Zeroth-Order Optimization loop...")
        trainer.train()
    
    if not training_args.no_eval:
        # =========================================================================
        # Evaluation
        # =========================================================================
        logger.info("Running evaluation and extracting metrics...")
        predict_results = trainer.predict(eval_dataset)
        metrics = predict_results.metrics
        
        dataset_name = getattr(data_args, 'data_path', 'scienceqa')
        metrics = evaluate_vlm_predictions(predict_results, dataset_name, metrics, logger)
        
        trainer.log(metrics)
        trainer.save_metrics("eval", metrics)

    if hasattr(model, 'original_forward'):
        model.forward = model.original_forward
    if "wandb" in training_args.report_to:
        wandb.finish()

if __name__ == "__main__":
    main()