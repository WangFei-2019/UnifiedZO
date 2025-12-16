# -*- coding: utf-8 -*-
# Benchmark using Refactored Trainer Classes directly with PEFT support
# Includes detailed Time Profiling (Perturb/Forward/Update) and Memory Statistics

import torch
import time
import numpy as np
import types
from fire import Fire
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# Import modules from the refactored code (ZO_all_code)
from arguments import ZOTrainingArguments
from trainer import get_trainer_class

# =============================================================================
# 1. Profiled Step Functions
# =============================================================================

def profiled_mezo_step(self, model, inputs, num_items_in_batch=None):
    """
    MeZO training step with detailed timing profiling.
    Replaces trainer.training_step
    """
    stats = {"perturb": 0.0, "forward": 0.0, "update": 0.0}
    
    # 1. Sample Seed
    self.zo_random_seed = np.random.randint(1000000000)
    
    # === Perturb (+) ===
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    self._perturb_mezo(scaling_factor=1)
    torch.cuda.synchronize()
    stats["perturb"] += time.perf_counter() - t0
    
    # === Forward (+) ===
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss1 = self.zo_forward(model, inputs)
    torch.cuda.synchronize()
    stats["forward"] += time.perf_counter() - t0
    
    # === Perturb (-) ===
    # From +1 to -1 requires -2 step
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    self._perturb_mezo(scaling_factor=-2)
    torch.cuda.synchronize()
    stats["perturb"] += time.perf_counter() - t0
    
    # === Forward (-) ===
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss2 = self.zo_forward(model, inputs)
    torch.cuda.synchronize()
    stats["forward"] += time.perf_counter() - t0
    
    # Calculate Projected Gradient Estimate
    self.projected_grad = (loss1 - loss2) / (2 * self.args.zo_eps)
    
    # === Restore Parameters ===
    # From -1 back to 0 requires +1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    self._perturb_mezo(scaling_factor=1)
    torch.cuda.synchronize()
    stats["perturb"] += time.perf_counter() - t0
    
    # === Update Parameters ===
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    self._update_mezo()
    torch.cuda.synchronize()
    stats["update"] += time.perf_counter() - t0
    
    # Save stats of this step to the trainer instance for external access
    self._last_step_stats = stats
    return loss1

# If LoZO or other methods are needed, add similar functions following the logic above
# Example: profiled_lozo_step ...

# =============================================================================
# Mocking Data
# =============================================================================

class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return {} 

# =============================================================================
# Main Benchmark Loop
# =============================================================================

def main_test(
        batch_size: int = 16,
        sequence_length: int = 256,
        method: str = "mezo", 
        peft_mode: str = "none", 
        steps: int = 10,
        load_float16: bool = True,
        model_path: str = "facebook/opt-1.3b"
):
    print(f"\n{'='*25} Benchmark Config {'='*25}")
    print(f"Method:     {method.upper()}")
    print(f"PEFT Mode:  {peft_mode.upper()}")
    print(f"Batch Size: {batch_size}")
    print(f"Steps:      {steps}")

    # 1. Args Setup
    args = ZOTrainingArguments(
        output_dir="./tmp_bench",
        per_device_train_batch_size=batch_size,
        trainer=method,
        use_cpu=False,
        zo_eps=1e-3,
        learning_rate=1e-6,
        # Method specifics
        zo_adamu_beta1=0.9, zo_adamu_beta2=0.999,
        lozo_rank=8, lozo_step_interval=100,
        hessian_smooth="constant1e-4", 
        sliding_window_length=14, momentum_fb_max=1.0,
        fzoo_n=4, adalezo_k_ratio=0.1,
    )

    if peft_mode == "lora":
        args.lora = True; args.lora_r = 8; args.lora_alpha = 16
    elif peft_mode == "prefix":
        args.prefix_tuning = True; args.num_prefix = 5
        args.prefix_init_by_real_act = False; args.reparam = False

    # 2. Load Model
    try:
        print(f"Loading model: {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16 if load_float16 else torch.float32)
    except Exception as e:
        print(f"Warning: Could not load {model_path}. Creating Dummy OPT-1.3B.")
        config = AutoConfig.from_pretrained("facebook/opt-1.3b")
        model = AutoModelForCausalLM.from_config(config)
        if load_float16: model = model.half()
        model = model.cuda()

    model.eval()

    # 3. PEFT Injection
    if peft_mode == "lora":
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0)
        model = get_peft_model(model, peft_config)
    elif peft_mode == "prefix":
        from tuners.prefix import PrefixTuning 
        PrefixTuning(model, num_prefix=args.num_prefix, reparam=args.reparam, float16=load_float16, init_by_real_act=args.prefix_init_by_real_act)

    # 4. Initialize Trainer
    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    tokenizer.pad_token = tokenizer.eos_token
    TrainerClass = get_trainer_class(args)
    
    trainer = TrainerClass(
        model=model, args=args,
        train_dataset=DummyDataset(length=100), tokenizer=tokenizer, data_collator=lambda x: x
    )

    # [FIX] Manually initialize optimizer/scheduler to avoid 'NoneType' error
    trainer.create_optimizer()
    trainer.create_scheduler(num_training_steps=steps)

    # [FIX] Monkey Patch: Inject the profiled training_step
    if method == "mezo":
        print("Injecting Profiled MeZO Step...")
        trainer.training_step = types.MethodType(profiled_mezo_step, trainer)
    else:
        print(f"Warning: Profiled step not implemented for {method}. Using default (no detailed breakdown).")
        # If needed, implement similar profiled_xxx_step for lozo, zo_adamu, etc. here

    # Forward wrappers setup
    if not hasattr(model, 'original_forward'): model.original_forward = model.forward
    from trainer.utils import forward_wrap_with_option_len
    model.forward = forward_wrap_with_option_len.__get__(model, type(model))

    # 5. Inputs
    dummy_inputs = {
        "input_ids": torch.randint(100, 1000, (batch_size, sequence_length), device=model.device),
        "attention_mask": torch.ones((batch_size, sequence_length), device=model.device),
        "labels": torch.randint(100, 1000, (batch_size, sequence_length), device=model.device),
        "option_len": torch.tensor([10] * batch_size, device=model.device)
    }

    # --- Start Benchmark ---
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    static_mem_gb = sum([torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]) / 1024**3
    print(f"Static Memory: {static_mem_gb:.4f} GB")

    # Accumulators
    stats_acc = {"perturb": [], "forward": [], "update": [], "peak_mem": [], "overhead": []}

    print(f"\n{'-'*105}")
    print(f"{'Step':<5} | {'Loss':<10} | {'Perturb(s)':<12} | {'Forward(s)':<12} | {'Update(s)':<12} | {'Peak(GB)':<10} | {'Overhead(MB)':<12}")
    print(f"{'-'*105}")

    # Warmup
    try:
        trainer.training_step(model, dummy_inputs)
    except Exception as e:
        print(f"Error in warmup: {e}")
        return

    for i in range(steps):
        # Reset memory stats
        torch.cuda.empty_cache()
        for d in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(d)
        
        torch.cuda.synchronize()
        
        # === Run Step ===
        loss = trainer.training_step(model, dummy_inputs)
        # ================
        
        torch.cuda.synchronize()
        
        # Get Time Stats (Fallback to 0 if not profiled)
        current_stats = getattr(trainer, "_last_step_stats", {"perturb": 0, "forward": 0, "update": 0})
        
        # Get Memory Stats
        peak_bytes = sum([torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())])
        peak_gb = peak_bytes / 1024**3
        overhead_mb = (peak_gb - static_mem_gb) * 1024

        # Print Row
        print(f"{i+1:<5} | {loss.item():<10.4f} | {current_stats['perturb']:<12.5f} | {current_stats['forward']:<12.5f} | {current_stats['update']:<12.5f} | {peak_gb:<10.4f} | {overhead_mb:<12.2f}")

        # Accumulate
        stats_acc["perturb"].append(current_stats['perturb'])
        stats_acc["forward"].append(current_stats['forward'])
        stats_acc["update"].append(current_stats['update'])
        stats_acc["peak_mem"].append(peak_gb)
        stats_acc["overhead"].append(overhead_mb)

    # Summary
    print(f"{'='*105}")
    print(f"AVERAGE STATISTICS ({method.upper()} + {peft_mode.upper()})")
    print(f"{'='*105}")
    print(f"Avg Perturb Time : {np.mean(stats_acc['perturb']):.5f} s")
    print(f"Avg Forward Time : {np.mean(stats_acc['forward']):.5f} s")
    print(f"Avg Update Time  : {np.mean(stats_acc['update']):.5f} s")
    print(f"{'-'*105}")
    total_time = np.mean(stats_acc['perturb']) + np.mean(stats_acc['forward']) + np.mean(stats_acc['update'])
    print(f"TOTAL STEP TIME  : {total_time:.5f} s")
    print(f"AVG PEAK MEMORY  : {np.mean(stats_acc['peak_mem']):.4f} GB")
    print(f"AVG OVERHEAD     : {np.mean(stats_acc['overhead']):.2f} MB")
    print(f"{'='*105}")

if __name__ == "__main__":
    Fire(main_test)
    # CUDA_VISIBLE_DEVICES=0 python test_zo_bench.py 16 256 mezo none 20 True /workspace/wangfei154/models/facebook/opt-13b