#!/bin/bash

# --- Defaults ---
MODEL=${MODEL:-google/vit-base-patch16-224}
MODEL_SHORT=(${MODEL//\// })
MODEL_SHORT="${MODEL_SHORT[-1]}"

TASK=${TASK:-uoft-cs/cifar10}
MODE=${MODE:-ft} # ft (Full Fine-Tuning) or lora

# Hyperparameters (Standard FT usually uses smaller LR)
BS=${BS:-64}
LR=${LR:-2e-5}      # Standard Fine-tuning LR
SEED=${SEED:-0}
STEPS=${STEPS:-10000} # FT usually converges faster than ZO
EVAL_STEPS=${EVAL_STEPS:-500}

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
    --train_as_classification True \
    --num_train 1000 \
    --num_eval 100 \
    --learning_rate $LR \
    --max_steps $STEPS \
    --logging_steps 10 \
    --save_steps $EVAL_STEPS \
    --eval_steps $EVAL_STEPS \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size 32 \
    --seed $SEED \
    --report_to wandb \
    --save_total_limit 1 \
    --save_model True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    $PEFT_ARGS \
    "$@"