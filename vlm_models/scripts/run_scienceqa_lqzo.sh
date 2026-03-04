#!/bin/bash

# ==============================================================================
# Academic Rigor: Configuration Management
# This script sets up the Low-Rank Zeroth-Order (LQZO) optimization for VLMs.
# It enforces variance reduction (Option-only loss) and deterministic execution.
# ==============================================================================

# Model and Data Configuration
MODEL_NAME="/workspace/wangfei154/models/llava-hf/llava-v1.5-13B-GPTQ" 
DATASET="scienceqa" # Options: "scienceqa" or "mathvista"
OUTPUT_DIR="./checkpoints/${DATASET}_lqzo_13b_academic"

# Experimental Settings
SEED=42
ZO_EPS=1e-3
LR=2e-5

echo "=================================================="
echo "Initiating Rigorous VLM Zeroth-Order Optimization (LQZO)"
echo "Target Model: $MODEL_NAME (13B GPTQ)"
echo "Benchmark: $DATASET"
echo "Global Seed: $SEED (For reproducible random perturbations)"
echo "=================================================="

# ------------------------------------------------------------------------------
# MANDATORY FLAGS ADDED:
# 1. --seed: Guarantee reproducibility of perturbation matrices.
# 2. --only_train_option True: CRITICAL. Evaluates ZO objective solely on the 
#    target response, masking out the extreme variance from prompt/image tokens.
# 3. --evaluation_strategy & --eval_steps: Monitor generalization dynamically.
# 4. --data_path: Dynamically switch between QA and reasoning benchmarks.
# ------------------------------------------------------------------------------

python run_vlm.py \
    --model_name_or_path "$MODEL_NAME" \
    --data_path "$DATASET" \
    --quant_method "gptq" \
    --freeze_vision_tower True \
    --freeze_mm_projector False \
    --output_dir "$OUTPUT_DIR" \
    --trainer "lqzo" \
    --zo_eps $ZO_EPS \
    --learning_rate $LR \
    --seed $SEED \
    --only_train_option True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 50 \
    --logging_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --bf16 True \
    --remove_unused_columns False \
    --report_to "wandb"