MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

MODE=${MODE:-ft}
BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-2000}

# LQZO Specific Defaults
QUANT_METHOD=${QUANT_METHOD:-gptq}
CLIP_GRAD=${CLIP_GRAD:-True}
LOZO_RANK=${LOZO_RANK:-2}
LOZO_INTERVAL=${LOZO_INTERVAL:-50}

if [ "$MODE" == "lora" ]; then
    LR=${LR:-1e-4}
    EPS=${EPS:-1e-2}
    PEFT_ARGS="--lora --lora_r 8 --lora_alpha 16"
elif [ "$MODE" == "prefix" ]; then
    LR=${LR:-1e-3}
    EPS=${EPS:-1e-1}
    PEFT_ARGS="--prefix_tuning --num_prefix 5 --prefix_init_by_real_act"
else
    PEFT_ARGS=""
fi

TAG=lqzo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-rank$LOZO_RANK
TASK_ARGS=""
GRAD_ACCUM_STEPS=1

case $TASK in
    CB)
        DEV=100
        ;;
    Copa)
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        BS=8
        GRAD_ACCUM_STEPS=2
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

# LQZO Specific Flags
EXTRA_ZO_ARGS="--quant_method $QUANT_METHOD --lozo_rank $LOZO_RANK --lozo_step_interval $LOZO_INTERVAL"
if [ "$CLIP_GRAD" = "True" ]; then
    EXTRA_ZO_ARGS="$EXTRA_ZO_ARGS --clip_zo_grad"
fi

echo "Running LQZO | Mode: $MODE | LR: $LR | EPS: $EPS | Model: $MODEL | Task: $TASK | Quant: $QUANT_METHOD | Rank: $LOZO_RANK"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result-lqzo/$TASK-${MODEL_NAME}-$TAG --tag $TAG \
    --train_set_seed $SEED --logging_steps 10 --max_steps $STEPS \
    --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --trainer lqzo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --per_device_eval_batch_size $BS \
    --lr_scheduler_type "constant" \
    --load_best_model_at_end --eval_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    $PEFT_ARGS \
    $TASK_ARGS \
    $EXTRA_ZO_ARGS \
    "$@"