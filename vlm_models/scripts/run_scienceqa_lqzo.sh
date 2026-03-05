#!/bin/bash

# Model and Data Configuration
MODEL_NAME="/workspace/wangfei154/models/llava-hf/llava-1.5-7b-hf" 
DATASET="scienceqa" # Options: "scienceqa" or "mathvista"
OUTPUT_DIR="./checkpoints/${DATASET}_lqzo_13b_academic"

# Experimental Settings
SEED=42
ZO_EPS=1e-3
LR=5e-7

echo "=================================================="
echo "Initiating Rigorous VLM Zeroth-Order Optimization (LQZO)"
echo "Target Model: $MODEL_NAME (13B GPTQ)"
echo "Benchmark: $DATASET"
echo "Global Seed: $SEED (For reproducible random perturbations)"
echo "=================================================="

python run_vlm.py \
    --model_name_or_path "$MODEL_NAME" \
    --data_path "$DATASET" \
    --quant_method "sim_quant" \
    --quantized_bit 4 \
    --freeze_vision_tower True \
    --freeze_mm_projector False \
    --output_dir "$OUTPUT_DIR" \
    --trainer "lqzo" \
    --zo_eps $ZO_EPS \
    --learning_rate $LR \
    --seed $SEED \
    --only_train_option True \v
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 20000 \
    --logging_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --bf16 True \
    --remove_unused_columns False \
    --report_to "wandb" \
    --quantize_llm True \
    --freeze_llm False \
    --quantize_vision False \
    --freeze_vision_tower True \
    --load_best_model_at_end True 