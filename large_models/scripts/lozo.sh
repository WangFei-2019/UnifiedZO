MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

MODE=${MODE:-ft}
BS=${BS:-16}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-2000}

# LoZO Specifics
LOZO_RANK=${LOZO_RANK:-2}
LOZO_INTERVAL=${LOZO_INTERVAL:-50}

if [ "$MODE" == "lora" ]; then
    LR=${LR:-5e-5}
    EPS=${EPS:-1e-2}
    PEFT_ARGS="--lora --lora_r 8 --lora_alpha 16"
elif [ "$MODE" == "prefix" ]; then
    LR=${LR:-1e-2}
    EPS=${EPS:-1e-1}
    PEFT_ARGS="--prefix_tuning --num_prefix 5 --prefix_init_by_real_act"
else
    LR=${LR:-1e-6}
    EPS=${EPS:-1e-3}
    PEFT_ARGS=""
fi

TAG=lozo-$MODE-$STEPS-$BS-$LR-$EPS-r${LOZO_RANK}-inv${LOZO_INTERVAL}-$SEED
TASK_ARGS=""
GRAD_ACCUM_STEPS=1

case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    # RTE)
    #     BS=8
    #     ;;
    # BoolQ)
    #     BS=8 # reduce batch size while extending training steps, equivalent training procedure
    #     ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
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
        # BS=8
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo "Running LOZO | Mode: $MODE | LR: $LR | Rank: $LOZO_RANK"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result-lozo/$TASK-${MODEL_NAME}-$TAG --tag $TAG \
    --train_set_seed $SEED --logging_steps 10 --max_steps $STEPS \
    --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --trainer lozo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS \
    --lr_scheduler_type "constant" \
    --load_best_model_at_end --eval_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --lozo_rank $LOZO_RANK --lozo_step_interval $LOZO_INTERVAL \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    $PEFT_ARGS \
    $TASK_ARGS \
    "$@"