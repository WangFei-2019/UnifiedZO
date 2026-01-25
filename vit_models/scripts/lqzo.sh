#!/bin/bash

MODEL_NAME="google/vit-base-patch16-224"
TASK_NAME="cifar10"
OUTPUT_DIR="result/vit_lqzo_${TASK_NAME}"

# --- Execution ---
# Run LQZO training.
# Key Arguments:
#   --lozo_rank: The rank 'r' for the low-rank decomposition.
#   --channel_scale: Scale factor for channel-wise reshaping in LQZO.
#   --momentum_lqzo: Enables specific momentum logic for the low-rank update.
python vit_models/run_vit.py \
    --model_name $MODEL_NAME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train 1000 \
    --num_eval 100 \
    --trainer lqzo \
    --learning_rate 1e-5 \
    --zo_eps 1e-3 \
    --lozo_rank 2 \
    --channel_scale 1 \
    --momentum_lqzo True \
    --quant_method none \
    --per_device_train_batch_size 16 \
    --train_as_classification True \
    --report_to wandb \
    --tag "lqzo_benchmark"