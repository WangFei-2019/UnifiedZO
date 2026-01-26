#!/bin/bash

# --- Defaults ---
MODEL=${MODEL:-google/vit-base-patch16-224}
MODEL_SHORT=(${MODEL//\// })
MODEL_SHORT="${MODEL_SHORT[-1]}"

TASK=${TASK:-uoft-cs/cifar10}
MODE=${MODE:-ft}

# QZO Specific Defaults
QUANT_METHOD=${QUANT_METHOD:-none} # Options: none (simulated), gptq, aqlm
ZO_SCALE=${ZO_SCALE:-1.0}          # Scale for step size parameters
CLIP_GRAD=${CLIP_GRAD:-True}       # Whether to clip ZO gradients

# Hyperparameters
BS=${BS:-64}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-1000}

# --- PEFT Logic ---
if [ "$MODE" == "lora" ]; then
    LR=${LR:-1e-4}
    PEFT_ARGS="--lora --lora_r 8 --lora_alpha 16"
else
    PEFT_ARGS=""
fi

# QZO Args Construction
EXTRA_ZO_ARGS="--quant_method $QUANT_METHOD --zo_scale $ZO_SCALE --train_unquantized True"
if [ "$CLIP_GRAD" = "True" ]; then
    EXTRA_ZO_ARGS="$EXTRA_ZO_ARGS --clip_zo_grad"
fi

# Generate Experiment Tag
TAG=qzo-${MODE}-${QUANT_METHOD}-${TASK}-lr${LR}-eps${EPS}
OUTPUT_DIR="result/vit_qzo/${TAG}"

echo "Running ViT QZO | Task: $TASK | Mode: $MODE | Quant: $QUANT_METHOD"

# --- Execution ---
python vit_models/run_vit.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $OUTPUT_DIR \
    --tag $TAG \
    --trainer qzo \
    --train_as_classification True \
    --num_train 1000 \
    --num_eval 100 \
    --learning_rate $LR \
    --zo_eps $EPS \
    --max_steps $STEPS \
    --logging_steps 10 \
    --save_steps $EVAL_STEPS \
    --eval_steps $EVAL_STEPS \
    --evaluation_strategy steps \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size 64 \
    --seed $SEED \
    --report_to wandb \
    --save_total_limit 1 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    $PEFT_ARGS \
    $EXTRA_ZO_ARGS \
    "$@"