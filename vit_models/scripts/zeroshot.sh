#!/bin/bash

# --- Defaults ---
MODEL=${MODEL:-google/vit-base-patch16-224}
MODEL_SHORT=(${MODEL//\// })
MODEL_SHORT="${MODEL_SHORT[-1]}"

TASK=${TASK:-uoft-cs/cifar10}

# Hyperparameters
BS=${BS:-32}        # Eval Batch Size
SEED=${SEED:-42}
NUM_EVAL=${NUM_EVAL:-1000} # Number of samples to evaluate

# Generate Experiment Tag
TAG=zeroshot-${TASK}-${MODEL_SHORT}
OUTPUT_DIR="result/vit_zeroshot/${TAG}"

echo "Running ViT Zero-Shot/Inference | Task: $TASK | Model: $MODEL"

# --- Execution ---
# trainer="none" skips the training loop and runs evaluation only
# num_train is set to 0 just to be explicit
python vit_models/run_vit.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $OUTPUT_DIR \
    --tag $TAG \
    --trainer none \
    --train_as_classification True \
    --num_train 0 \
    --num_eval $NUM_EVAL \
    --per_device_eval_batch_size $BS \
    --seed $SEED \
    --report_to wandb \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    "$@"