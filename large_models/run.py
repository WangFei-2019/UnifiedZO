import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging
import json
import random
import time
from tqdm import tqdm
import numpy as np
from dataclasses import asdict
import wandb

from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    HfArgumentParser, 
    DataCollatorForTokenClassification,
    set_seed
)
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, TaskType, PromptTuningConfig, PromptTuningInit

# --- Local Imports ---
# Ensure these files are in the same directory or properly installed in PYTHONPATH
from arguments import ZOTrainingArguments
from tasks import get_task
from zo_core.trainer import get_trainer_class, BaseZOTrainer
from utils import (
    count_time,
    write_metrics_to_file,
    DataCollatorWithPaddingAndNesting,
    NondiffCollator,
    process_dataset, 
    result_file_tag
)
from evaluation import Evaluator
# from evaluation import BatchedEvaluator as Evaluator

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    # 1. Parse Arguments
    parser = HfArgumentParser((ZOTrainingArguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    if "wandb" in args.report_to and args.local_rank <= 0:
        wandb.init(
            project=f"ZO_{args.trainer}",
            name=f"{args.task_name}-{args.model_name.split('/')[-1]}-{args.tag if args.tag else ''}",
            config=asdict(args)
        )

    # Print args for debugging
    if args.local_rank <= 0:
        print(f"Training Arguments: {args}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # 2. Setup Task
    # Load the specific task logic (template, data loading)
    task = get_task(args.task_name)
    
    # Sample training sets (supports Few-Shot sampling logic)
    # If num_train is huge, this essentially loads the full dataset
    train_sets = task.sample_train_sets(
        num_train=args.num_train, 
        num_dev=args.num_dev, 
        num_eval=args.num_eval, 
        num_train_sets=args.num_train_sets, 
        seed=args.train_set_seed
    )

    # 3. Load Model and Tokenizer
    logger.info(f"Loading model: {args.model_name}")
    
    with count_time("Loading Model & Tokenizer"):
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        
        # Untie embeddings if requested (rarely used for modern LMs, but legacy support)
        if args.untie_emb:
            logger.warning("Untie embeddings and LM head")
            config.tie_word_embeddings = False

        # Load Tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True, clean_up_tokenization_spaces=True) # # In the MeZO implementation (with the specified transformers version), the default value of clean_up_tokenization_spaces is True.

        tokenizer.padding_side = "left"

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        config.pad_token_id = tokenizer.pad_token_id
        
        # Apply Tokenizer fixes for certain models
        if "opt" in args.model_name:
            tokenizer.bos_token_id = 0

        # if "llama" in args.model_name.lower() or "mistral" in args.model_name.lower():
        #     # LLaMA/Mistral usually don't have a default pad token
        #     if tokenizer.pad_token_id is None:
        #         tokenizer.pad_token_id = 0  # <unk> or often set to eos_token_id

        # Determine Torch Data Type
        if args.load_float32:
            torch_dtype = torch.float32
        elif args.load_float16:
            torch_dtype = torch.float16
        elif args.load_bfloat16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = "auto"

        # Initialize Model
        # device_map='auto' handles model placement (CPU/GPU)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            device_map='auto' if not args.no_auto_device else None,
            torch_dtype=torch_dtype, 
            load_in_8bit=args.load_int8,
            trust_remote_code=True
        )
        model.eval()

    # 4. Inject Parameter Efficient Fine-Tuning (PEFT)
    # We support LoRA and Prefix Tuning via custom implementations or standard libraries
    task_type = TaskType.CAUSAL_LM
    if args.lora:
        logger.info("Injecting LoRA via PEFT...")
        peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            # target_modules=["q_proj", "v_proj"] 
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # elif args.prefix_tuning:
    #     logger.info("Injecting Prefix Tuning via PEFT...")
    #     peft_config = PrefixTuningConfig(
    #         task_type=task_type,
    #         num_virtual_tokens=args.num_prefix,
    #         prefix_projection=args.reparam,
    #         encoder_hidden_size=model.config.hidden_size

    #     )
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()
    elif args.prefix_tuning:
        logger.info("Injecting Prefix Tuning via Custom MeZO Implementation...")
        from zo_core.tuners import PrefixTuning 
        PrefixTuning(
            model, 
            num_prefix=args.num_prefix, 
            reparam=args.reparam, 
            float16=args.load_float16, 
            init_by_real_act=args.prefix_init_by_real_act
        )
        
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    elif args.prompt_tuning:
        logger.info("Injecting Prompt Tuning via PEFT...")

        tuning_init = PromptTuningInit.RANDOM
        init_text = None
        
        if args.prompt_init_by_real_tokens:
            tuning_init = PromptTuningInit.TEXT

            random_ids = torch.randint(
                low=0, 
                high=tokenizer.vocab_size, 
                size=(args.num_virtual_tokens,)
            ).tolist()
            
            init_text = tokenizer.decode(random_ids, skip_special_tokens=True)
            logger.info(f"Initialized Prompt with text: {init_text}")

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.num_virtual_tokens,
            prompt_tuning_init=tuning_init,
            prompt_tuning_init_text=init_text,
            tokenizer_name_or_path=args.model_name,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    elif args.head_tuning:
        logger.info("Injecting Head Tuning via Custom Implementation...")
        from zo_core.tuners import HeadTuning
        HeadTuning(model, args)

    # 5. Training Loop / Evaluation Loop
    # Handles cases where we might have multiple training sets (few-shot variance)
    
    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Loop over training sets
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample Eval Data
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            # --- Split Dev Data from Train (if needed) ---
            dev_samples = None
            if args.num_dev is not None:
                dev_samples = train_samples[-args.num_dev:]
                train_samples = train_samples[:-args.num_dev]

            # --- Data Processing ---
            logger.info(f"Processing data for train set {train_set_id}...")
            train_dataset = process_dataset(args, task, train_samples, tokenizer, is_training=True)
            # Set is_training=True to include gold answers in the encoding. 
            # This ensures 'option_len' is non-zero, allowing the Trainer to calculate a valid 'eval_loss' (perplexity) 
            # instead of NaN. Note: This only affects the Trainer's loss logging; actual evaluation metrics (e.g., F1, EM) 
            # are computed independently by the Evaluator and remain unaffected.
            eval_dataset = process_dataset(args, task, eval_samples, tokenizer, is_training=True)
            
            # --- Model Patching for ZO ---
            # We replace the forward pass to support calculating loss only on the 'option' part.
            # This is crucial for ZO to optimize the correct objective.
            if args.only_train_option and not args.non_diff:
                if not hasattr(model, 'original_forward'):
                    model.original_forward = model.forward

                if args.trainer == "pzo" or args.trainer == "adapzo":
                    if args.logits:
                        from zo_core.trainer.utils import forward_wrap_with_option_len_pzo_logits
                        model.forward = forward_wrap_with_option_len_pzo_logits.__get__(model, type(model))
                    else:
                        from zo_core.trainer.utils import  forward_wrap_with_option_len_pzo
                        model.forward = forward_wrap_with_option_len_pzo.__get__(model, type(model))
                elif args.trainer == "fzoo" or args.trainer == "adafzoo":
                    from zo_core.trainer.utils import forward_wrap_with_option_len_fzoo
                    model.forward = forward_wrap_with_option_len_fzoo.__get__(model, type(model))
                else:
                    # MeZO, LoZO, HiZOO use ZO wrapper
                    from zo_core.trainer.utils import forward_wrap_with_option_len
                    model.forward = forward_wrap_with_option_len.__get__(model, type(model))

            # --- Select Collator ---
            if args.non_diff:
                collator = NondiffCollator(tokenizer, pad_to_multiple_of=8)
            elif args.train_as_classification:
                collator = DataCollatorWithPaddingAndNesting(tokenizer, pad_to_multiple_of=8)
            else:
                collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

            # --- Initialize ZO Trainer via Factory ---
            TrainerClass = get_trainer_class(args)
            logger.info(f"Initializing Trainer Class: {TrainerClass.__name__}")
            
            trainer = TrainerClass(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=collator,
                zo_evaluator=Evaluator(args, task, tokenizer, model),
                raw_dev_samples=dev_samples,
                raw_test_samples=eval_samples
            )

            # --- Train ---
            if args.trainer != "none":
                logger.info("Starting Training...")

                # Linear Probing Branch
                if args.linear_probing:
                    from zo_core.tuners import perform_linear_probing
                    # Call independent module, skipping standard trainer.train()
                    perform_linear_probing(args, model, tokenizer, train_dataset, collator)
                    logger.info("Linear Probing complete. Skipping standard ZO training loop.")
                    
                    # Save model (LP modifies LM Head weights, so saving is necessary)
                        # if args.save_model:
                        #      trainer.save_model()
                else:
                    # Standard ZO/Fine-tuning Training
                    train_result = trainer.train()
                    
                    # Log training results
                    if args.local_rank <= 0:
                        logger.info(f"Training finished. Loss: {train_result.training_loss}")
                    
                # Save Model
                if args.save_model:
                    logger.info("Saving model...")
                    trainer.save_model(output_dir=os.path.join(args.output_dir, "final_model"))

            # --- Evaluate ---
            if not args.no_eval:
                logger.info("Starting Evaluation...")
                
                # Initialize custom Evaluator
                evaluator = Evaluator(args, task, tokenizer, trainer.model)
                metrics = evaluator.evaluate(eval_samples, [])

                # Add dev metrics if available
                if dev_samples is not None:
                    # We need to process dev samples into a dataset temporarily
                    dev_metrics = evaluator.evaluate(dev_samples, [])
                    for m in dev_metrics:
                        metrics["dev_" + m] = dev_metrics[m]

                # Log to console
                logger.info(f"Evaluation Metrics: {metrics}")
                
                # Write to file
                if args.local_rank <= 0:
                    # Construct output filename
                    if args.result_file:
                        output_path = args.result_file
                    else:
                        fname = result_file_tag(args) + f"-trainset{train_set_id}.json"
                        output_path = os.path.join(args.output_dir, fname)
                        
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    write_metrics_to_file(metrics, output_path)
                    logger.info(f"Metrics written to {output_path}")

            # --- Clean up Model for next iteration ---
            # Restore the original forward method to avoid nesting wrappers if we loop again
            if hasattr(model, "original_forward"):
                model.forward = model.original_forward
                del model.original_forward

    else:
        # --- Zero-Shot / few-shot / Inference Only Mode ---
        # Used for standard ICL evaluation where we have one training set per eval sample (handled internally by task sampling logic usually)
        # Or pure Zero-shot.
        
        logger.info("Running in Inference-Only / One-Train-Set-Per-Eval mode")
        assert args.trainer == "none", "Trainer must be 'none' for this mode."
        
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        evaluator = Evaluator(args, task, tokenizer, model)

        # metrics = evaluator.evaluate(eval_samples=eval_samples, train_samples=train_sets, one_train_set_per_eval_sample=True, verbose_len=3)
        metrics = evaluator.evaluate(eval_samples=eval_samples, train_samples=train_sets, one_train_set_per_eval_sample=True)
        if args.local_rank <= 0:
            fname = result_file_tag(args) + "-inference.json"
            output_path = os.path.join(args.output_dir, fname)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_metrics_to_file(metrics, output_path)

    logger.info(metrics)
    if args.report_to is not None and "wandb" in args.report_to:
        wandb.log({"result/" + key:value for key, value in metrics.items()})
        
    if args.report_to == "wandb" and args.local_rank <= 0:
        wandb.finish()


if __name__ == "__main__":
    # # --------------------------------------
    # import debugpy

    # # 启动调试服务器，监听本地主机的5678端口
    # debugpy.listen(('localhost', 15678))
    # print("Waiting for debugger attach...")

    # # 可选：暂停程序，直到调试器附加
    # debugpy.wait_for_client()

    # # 继续你的代码
    # print("Debugger attached, continuing execution")
    # # ---------------------------------------
    main()