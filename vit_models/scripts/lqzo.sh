#!/bin/bash

# --- Defaults ---
MODEL=${MODEL:-google/vit-base-patch16-224}
MODEL_SHORT=(${MODEL//\// })
MODEL_SHORT="${MODEL_SHORT[-1]}"

TASK=${TASK:-uoft-cs/cifar10}
MODE=${MODE:-ft}

# LQZO Specific Defaults
LOZO_RANK=${LOZO_RANK:-2}
CHANNEL_SCALE=${CHANNEL_SCALE:-1}
MOMENTUM=${MOMENTUM:-False}

TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-100}

# Hyperparameters
BS=${BS:-64}        # Batch Size
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-2000}

# --- PEFT Logic ---
if [ "$MODE" == "lora" ]; then
    LR=${LR:-1e-4}
    PEFT_ARGS="--lora --lora_r 8 --lora_alpha 16"
else
    PEFT_ARGS=""
fi

# LQZO Args Construction
EXTRA_ZO_ARGS="--lozo_rank $LOZO_RANK --channel_scale $CHANNEL_SCALE"
if [ "$MOMENTUM" = "True" ]; then
    EXTRA_ZO_ARGS="$EXTRA_ZO_ARGS --momentum_lqzo"
fi

# Generate Experiment Tag
TAG=lqzo-${MODE}-rank${LOZO_RANK}-${TASK}-lr${LR}-eps${EPS}
OUTPUT_DIR="result/vit_lqzo/${TAG}"

echo "Running ViT LQZO | Task: $TASK | Rank: $LOZO_RANK | Momentum: $MOMENTUM"

# --- Execution ---
python vit_models/run_vit.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $OUTPUT_DIR \
    --tag $TAG \
    --trainer lqzo \
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
    $EXTRA_ZO_ARGS \
    "$@"