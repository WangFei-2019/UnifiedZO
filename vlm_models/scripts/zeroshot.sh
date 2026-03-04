#!/bin/bash
MODEL=${MODEL:-"/workspace/wangfei154/models/llava-hf/llava-1.5-7b-hf"}
DATASET=${DATASET:-"scienceqa"}

BS=${BS:-8}        
SEED=${SEED:-42}
DEV=${DEV:-1000}
EVAL=${EVAL:--1}

QUANT_METHOD=${QUANT_METHOD:-"none"}

MODEL_SHORT=$(basename $MODEL)
TAG=zeroshot-${DATASET}-${MODEL_SHORT}
OUTPUT_DIR="result/vlm_zeroshot/${TAG}"

echo "=================================================="
echo "Running VLM Zero-Shot/Baseline Inference"
echo "Task: $DATASET | Model: $MODEL"
echo "Quantization: $QUANT_METHOD"
echo "=================================================="

python run_vlm.py \
    --model_name_or_path "$MODEL" \
    --data_path "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --trainer none \
    --quant_method "$QUANT_METHOD" \
    --freeze_vision_tower True \
    --freeze_mm_projector True \
    --freeze_llm True \
    --num_dev $DEV \
    --num_eval $EVAL \
    --per_device_eval_batch_size $BS \
    --seed $SEED \
    --report_to wandb \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    "$@"