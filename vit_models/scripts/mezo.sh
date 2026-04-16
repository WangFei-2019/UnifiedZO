#!/bin/bash

# --- Defaults ---
MODEL=${MODEL:-google/vit-base-patch16-224}
# Extract short name for tagging (e.g., vit-base-patch16-224)
MODEL_SHORT=(${MODEL//\// })
MODEL_SHORT="${MODEL_SHORT[-1]}"

TASK=${TASK:-uoft-cs/cifar10}
MODE=${MODE:-ft} # ft (Full Tuning) or lora

TRAIN=${TRAIN:--1}
DEV=${DEV:-1000}
EVAL=${EVAL:--1}

# Hyperparameters
BS=${BS:-64}        # Batch Size
LR=${LR:-1e-5}      # Learning Rate
EPS=${EPS:-1e-3}    # ZO Epsilon
SEED=${SEED:-0}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-2000}

# --- PEFT / Mode Logic ---
if [ "$MODE" == "lora" ]; then
    # LoRA specific defaults if not overridden
    LR=${LR:-1e-4}
    PEFT_ARGS="--lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1"
else
    PEFT_ARGS=""
fi

# Generate Experiment Tag
TAG=mezo-${MODE}-${TASK}-${MODEL_SHORT}-lr${LR}-eps${EPS}-seed${SEED}
OUTPUT_DIR="result/vit_mezo/${TAG}"

echo "Running ViT MeZO | Task: $TASK | Mode: $MODE | Model: $MODEL"
echo "Output Dir: $OUTPUT_DIR"

# --- Execution ---
python vit_models/run_vit.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $OUTPUT_DIR \
    --tag $TAG \
    --trainer mezo \
    --train_as_classification True \
    --num_train $TRAIN \
    --num_dev $DEV \
    --num_eval $EVAL \
    --learning_rate $LR \
    --zo_eps $EPS \
    --max_steps $STEPS \
    --logging_steps 10 \
    --save_steps $EVAL_STEPS \
    --eval_steps $EVAL_STEPS \
    --eval_strategy steps \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size 64 \
    --seed $SEED \
    --report_to wandb \
    --save_total_limit 1 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --load_best_model_at_end True \
    --metric_for_best_model eval_validation_loss \
    $PEFT_ARGS \
    "$@"