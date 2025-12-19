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

# --- AdaLeZO Params ---
ADA_KRATIO=${ADA_KRATIO:-0.1}       # Ratio of layers to select (e.g., 0.1 for 10%)
ADA_TAU=${ADA_TAU:-0.1}             # Temperature for Softmax/Gumbel distribution in layer selection
ADA_C=${ADA_C:-0.7}                 # Exploration constant for Bandit (UCB)
ADA_CLIP=${ADA_CLIP:-10}            # Clipping threshold for Inverse Probability Weighting (IPW)

# --- DiZO Specific Params ---
DIZO_INTERVAL=${DIZO_INTERVAL:-100}     # Interval steps for projection
DIZO_ITERS=${DIZO_ITERS:-5}             # Number of ZO optimization steps for Gamma per projection
ZO_EPS_PROJ=${ZO_EPS_PROJ:-1e-3}
STEP_SIZE_PROJ=${STEP_SIZE_PROJ:-0.1}
CLIP_RANGE=${CLIP_RANGE:-1e-4}

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

TAG=adadizo-$MODE-$STEPS-$BS-$LR-k${ADA_KRATIO}-int${DIZO_INTERVAL}-$SEED
TASK_ARGS=""
GRAD_ACCUM_STEPS=1

case $TASK in
    CB) DEV=100 ;;
    Copa) DEV=100; TASK_ARGS="--train_as_classification False" ;;
    ReCoRD) TASK_ARGS="--train_as_classification False" ;;
    DROP) BS=8; GRAD_ACCUM_STEPS=2; TASK_ARGS="--train_as_classification False" ;;
    SQuAD) TASK_ARGS="--train_as_classification False" ;;
esac

echo "Running AdaDiZO | Mode: $MODE | LR: $LR | K: $ADA_KRATIO | Interval: $DIZO_INTERVAL | Task: $TASK"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result-adadizo/$TASK-${MODEL_NAME}-$TAG --tag $TAG \
    --train_set_seed $SEED --logging_steps 10 --max_steps $STEPS \
    --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --trainer adadizo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --per_device_eval_batch_size $BS \
    --lr_scheduler_type "constant" \
    --load_best_model_at_end --eval_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    \
    --adalezo_k_ratio $ADA_KRATIO \
    --adalezo_tau $ADA_TAU \
    --adalezo_c $ADA_C \
    --adalezo_ipw_clip $ADA_CLIP \
    \
    --dizo_interval $DIZO_INTERVAL \
    --dizo_iters $DIZO_ITERS \
    --zo_eps_projection $ZO_EPS_PROJ \
    --step_size_projection $STEP_SIZE_PROJ \
    --clip_range $CLIP_RANGE \
    \
    $PEFT_ARGS \
    $TASK_ARGS \
    "$@"