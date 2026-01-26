#!/bin/bash

# --- Defaults ---
MODEL=${MODEL:-google/vit-base-patch16-224}
MODEL_SHORT=(${MODEL//\// })
MODEL_SHORT="${MODEL_SHORT[-1]}"

TASK=${TASK:-uoft-cs/cifar10}
MODE=${MODE:-ft} # ft (Full Fine-Tuning) or lora

TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}

# Hyperparameters (Standard FT usually uses smaller LR)
BS=${BS:-64}
LR=${LR:-1e-4}      # Standard Fine-tuning LR
SEED=${SEED:-0}
EPOCH=${EPOCH:-5}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}

# --- PEFT Logic ---
if [ "$MODE" == "lora" ]; then
    LR=${LR:-1e-4} # LoRA often handles higher LR
    PEFT_ARGS="--lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1"
else
    PEFT_ARGS=""
fi

# Generate Experiment Tag
TAG=ft-${MODE}-${TASK}-${MODEL_SHORT}-lr${LR}-seed${SEED}
OUTPUT_DIR="result/vit_ft/${TAG}"

echo "Running ViT Fine-Tuning (BP) | Task: $TASK | Mode: $MODE | Model: $MODEL"

# --- Execution ---
# trainer="regular" triggers standard backpropagation (First-Order)
python vit_models/run_vit.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $OUTPUT_DIR \
    --tag $TAG \
    --trainer regular \
    --lr_scheduler_type linear \
    --train_as_classification True \
    --num_train $TRAIN \
    --num_dev $DEV \
    --num_eval $EVAL \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --eval_strategy epoch \
    --save_strategy epoch \
    --logging_steps 1 \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size 64 \
    --seed $SEED \
    --report_to wandb \
    --save_total_limit 1 \
    --save_model True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --load_best_model_at_end True \
    --metric_for_best_model eval_validation_loss \
    $PEFT_ARGS \
    "$@"