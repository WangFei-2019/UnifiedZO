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

# --- DiZO Specific Params ---
DIZO_INTERVAL=${DIZO_INTERVAL:-100}     # Interval steps for projection
DIZO_ITERS=${DIZO_ITERS:-10}             # Number of ZO optimization steps for Gamma per projection
ZO_EPS_PROJ=${ZO_EPS_PROJ:-0.1}        # Epsilon for ZO gradient estimation during projection
STEP_SIZE_PROJ=${STEP_SIZE_PROJ:-2.0}   # Step size (learning rate) for Gamma optimization
CLIP_RANGE=${CLIP_RANGE:-0.2}          # Clipping range (tau) for constraint parameters
NORM_MODE=${NORM_MODE:-l2}              # Norm mode for constraints (l2 or mars)

if [ "$MODE" == "lora" ]; then
    LR=${LR:-5e-5}
    EPS=${EPS:-1e-2}
    PEFT_ARGS="--lora --lora_r 8 --lora_alpha 16"
elif [ "$MODE" == "prefix" ]; then
    LR=${LR:-1e-3}
    EPS=${EPS:-1e-1}
    PEFT_ARGS="--prefix_tuning --num_prefix 5 --prefix_init_by_real_act"
else
    LR=${LR:-1e-6}
    EPS=${EPS:-1e-3}
    PEFT_ARGS=""
fi

TAG=dizo-$MODE-$STEPS-$BS-$LR-$EPS-int${DIZO_INTERVAL}-it${DIZO_ITERS}-$SEED
TASK_ARGS=""
GRAD_ACCUM_STEPS=1

case $TASK in
    CB) DEV=100 ;;
    Copa) DEV=100; TASK_ARGS="--train_as_classification False" ;;
    ReCoRD) TASK_ARGS="--train_as_classification False" ;;
    DROP) BS=8; GRAD_ACCUM_STEPS=2; TASK_ARGS="--train_as_classification False" ;;
    SQuAD) TASK_ARGS="--train_as_classification False" ;;
esac

echo "Running DiZO | Mode: $MODE | LR: $LR | EPS: $EPS | Interval: $DIZO_INTERVAL | Model: $MODEL | Task: $TASK"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result-dizo/$TASK-${MODEL_NAME}-$TAG --tag $TAG \
    --train_set_seed $SEED --logging_steps 10 --max_steps $STEPS \
    --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --trainer dizo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --per_device_eval_batch_size $BS \
    --lr_scheduler_type "linear" \
    --load_best_model_at_end --eval_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    \
    --dizo_interval $DIZO_INTERVAL \
    --dizo_iters $DIZO_ITERS \
    --zo_eps_projection $ZO_EPS_PROJ \
    --step_size_projection $STEP_SIZE_PROJ \
    --clip_range $CLIP_RANGE \
    --norm_mode $NORM_MODE \
    \
    $PEFT_ARGS \
    $TASK_ARGS \
    "$@"