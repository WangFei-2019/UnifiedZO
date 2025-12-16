# -*- coding: utf-8 -*-
# Benchmark using Refactored Trainer Classes directly with PEFT support
# Requires: ZO_all_code structure (large_models package) and peft library

import torch
import time
import numpy as np
from fire import Fire
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerFast
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# Import modules from the refactored code (ZO_all_code)
from arguments import ZOTrainingArguments
from trainer import get_trainer_class

# =============================================================================
# Mocking Data
# =============================================================================

class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        # Trainer handles batching internally; return empty dict here
        return {} 

# =============================================================================
# Main Benchmark Loop
# =============================================================================

def main_test(
        batch_size: int = 16,
        sequence_length: int = 256,
        method: str = "mezo",  # Options: mezo, zoadamu, lozo, hizoo, adalezo, pzo, fzoo
        peft_mode: str = "none", # Options: none, lora, prefix
        steps: int = 10,
        load_float16: bool = True,
        model_path: str = "facebook/opt-1.3b" # Or use a dummy path to trigger dummy model creation
):
    print(f"\n{'='*25} Benchmark Config {'='*25}")
    print(f"Method:     {method.upper()}")
    print(f"PEFT Mode:  {peft_mode.upper()}")
    print(f"Batch Size: {batch_size}")
    print(f"Seq Length: {sequence_length}")
    print(f"Steps:      {steps}")

    # 1. Construct Training Arguments
    # Set arguments based on method and peft_mode to ensure Trainer initializes internal state correctly
    args = ZOTrainingArguments(
        output_dir="./tmp_bench",
        per_device_train_batch_size=batch_size,
        trainer=method,
        use_cpu=False,
        logging_steps=100, # Reduce logging overhead
        # General ZO parameters
        zo_eps=1e-3,
        learning_rate=1e-6,
        # Method-specific parameters (Default values from ZO_all_code)
        zo_adamu_beta1=0.9,
        zo_adamu_beta2=0.999,
        lozo_rank=8,
        lozo_step_interval=100,
        hessian_smooth="constant1e-4", # Hizoo
        sliding_window_length=14, # PseuZO
        momentum_fb_max=1.0,      # PseuZO
        fzoo_n=4,                 # FZOO
        adalezo_k_ratio=0.1,      # AdaLeZO
    )

    # Set flags for PEFT args to ensure compatibility with Trainer logic
    if peft_mode == "lora":
        args.lora = True
        args.lora_r = 8
        args.lora_alpha = 16
    elif peft_mode == "prefix":
        args.prefix_tuning = True
        args.num_prefix = 5
        args.prefix_init_by_real_act = False # Disable for dummy model stability
        args.reparam = False # ZO usually doesn't use reparam

    # 2. Load Model (Use Dummy or Real)
    try:
        print(f"Loading model: {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16 if load_float16 else torch.float32
        )
    except Exception as e:
        print(f"Warning: Could not load {model_path} ({e}). Creating Dummy OPT-1.3B.")
        config = AutoConfig.from_pretrained("facebook/opt-1.3b")
        # Ensure parameter count is realistic for memory testing
        model = AutoModelForCausalLM.from_config(config)
        if load_float16: model = model.half()
        model = model.cuda()

    model.eval() # ZO usually runs in eval mode (except PseuZO, handled by Trainer)

    # 3. Apply PEFT (LoRA or Prefix)
    if peft_mode == "lora":
        print("Injecting LoRA via PEFT library...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            # target_modules default to query/value for OPT/Llama usually
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    elif peft_mode == "prefix":
        print("Injecting Prefix Tuning via Custom MeZO Implementation...")
        # Importing from the local refactored structure
        from large_models.tuners.prefix import PrefixTuning 
        PrefixTuning(
            model, 
            num_prefix=args.num_prefix, 
            reparam=args.reparam, 
            float16=load_float16, 
            init_by_real_act=args.prefix_init_by_real_act
        )
        # Log trainable parameters manually for Prefix
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")

    # 4. Initialize Trainer
    # We need a dummy tokenizer and collator to satisfy Trainer initialization checks
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2") 
    tokenizer.pad_token = tokenizer.eos_token
    
    # Get corresponding Trainer class
    TrainerClass = get_trainer_class(args)
    print(f"Initializing {TrainerClass.__name__}...")

    trainer = TrainerClass(
        model=model,
        args=args,
        train_dataset=DummyDataset(length=100), # Dummy dataset
        tokenizer=tokenizer,
        data_collator=lambda x: x # Dummy collator
    )

    # Manually attach forward wrappers if needed (usually handled in run.py)
    # This is crucial for PseuZO/FZOO to work correctly
    if method == "pzo":
        from large_models.trainer.utils import forward_wrap_with_option_len_pzo
        model.forward = forward_wrap_with_option_len_pzo.__get__(model, type(model))
    elif method == "fzoo":
        from large_models.trainer.utils import forward_wrap_with_option_len_fzoo
        model.forward = forward_wrap_with_option_len_fzoo.__get__(model, type(model))
    else:
        # MeZO, LoZO, HiZOO, AdaLeZO use the standard ZO wrapper
        from large_models.trainer.utils import forward_wrap_with_option_len
        model.forward = forward_wrap_with_option_len.__get__(model, type(model))

    # 5. Construct Input Data
    # Construct tensor dict directly to bypass DataLoader overhead and focus on step performance
    dummy_inputs = {
        "input_ids": torch.randint(100, 1000, (batch_size, sequence_length), device=model.device),
        "attention_mask": torch.ones((batch_size, sequence_length), device=model.device),
        "labels": torch.randint(100, 1000, (batch_size, sequence_length), device=model.device) 
    }
    # Add option_len (Required by ZO forward wrappers)
    # Assuming the last 10 tokens are the "option" (candidate) part
    dummy_inputs["option_len"] = [10] * batch_size 

    # --- Start Benchmark ---

    # Measure Initial Static Memory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    static_mem_gb = sum([torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]) / 1024**3
    print(f"Static Memory (Model + Adapter): {static_mem_gb:.4f} GB")

    print(f"\n{'-'*85}")
    print(f"{'Step':<5} | {'Step Time (s)':<15} | {'Peak Mem (GB)':<15} | {'Overhead (MB)':<15}")
    print(f"{'-'*85}")

    latencies = []
    peak_mems = []
    overheads = []

    # Warmup Step
    print("Warmup step...")
    try:
        trainer.training_step(model, dummy_inputs)
    except Exception as e:
        print(f"Error in warmup: {e}")
        # Ensure dummy_inputs structure matches what the model expects
        return

    # Measurement Loop
    for i in range(steps):
        # Reset memory stats
        torch.cuda.empty_cache()
        for d in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(d)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # === Core Call ===
        # Directly call Trainer.training_step
        # This includes: Perturb -> Forward -> Update (and any internal hook overhead)
        loss = trainer.training_step(model, dummy_inputs)
        # =================
        
        torch.cuda.synchronize()
        t_step = time.perf_counter() - t0
        
        peak_bytes = sum([torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())])
        peak_gb = peak_bytes / 1024**3
        overhead_mb = (peak_gb - static_mem_gb) * 1024

        latencies.append(t_step)
        peak_mems.append(peak_gb)
        overheads.append(overhead_mb)

        print(f"{i+1:<5} | {t_step:<15.5f} | {peak_gb:<15.4f} | {overhead_mb:<15.2f}")

    # Summary
    print(f"{'='*85}")
    print(f"AVERAGE ({method.upper()} + {peft_mode.upper()})")
    print(f"Step Time : {np.mean(latencies):.5f} s")
    print(f"Peak Mem  : {np.mean(peak_mems):.4f} GB")
    print(f"Overhead  : {np.mean(overheads):.2f} MB")
    print(f"{'='*85}")

if __name__ == "__main__":
    Fire(main_test)
    # CUDA_VISIBLE_DEVICES=0 python test_zo_bench.py 16 256 mezo none 20 True /workspace/wangfei154/models/facebook/opt-13b