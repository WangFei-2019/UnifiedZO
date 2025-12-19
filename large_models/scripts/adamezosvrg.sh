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
ADA_KRATIO=${ADA_KRATIO:-0.2}       # Ratio of layers to select (e.g., 0.2 for 20%)
ADA_TAU=${ADA_TAU:-0.1}             # Temperature for Softmax exploration
ADA_C=${ADA_C:-0.7}                 # Exploration constant (UCB)
ADA_CLIP=${ADA_CLIP:-10}            # IPW clipping threshold

# --- MeZO-SVRG Params ---
SVRG_Q=${SVRG_Q:-100}   # Frequency to update the anchor model
SVRG_K=${SVRG_K:-1}     # Samples for anchor gradient estimation

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

TAG=adamezosvrg-$MODE-$STEPS-$BS-$LR-k${ADA_KRATIO}-q${SVRG_Q}-$SEED
TASK_ARGS=""
GRAD_ACCUM_STEPS=1

case $TASK in
    CB) DEV=100 ;;
    Copa) DEV=100; TASK_ARGS="--train_as_classification False" ;;
    ReCoRD) TASK_ARGS="--train_as_classification False" ;;
    DROP) BS=8; GRAD_ACCUM_STEPS=2; TASK_ARGS="--train_as_classification False" ;;
    SQuAD) TASK_ARGS="--train_as_classification False" ;;
esac

echo "Running AdaMeZO-SVRG | K: $ADA_KRATIO | Q: $SVRG_Q | Task: $TASK"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result-adamezosvrg/$TASK-${MODEL_NAME}-$TAG --tag $TAG \
    --train_set_seed $SEED --logging_steps 10 --max_steps $STEPS \
    --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --trainer adamezosvrg --load_float16 \
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
    --svrg_q $SVRG_Q \
    --svrg_k $SVRG_K \
    \
    $PEFT_ARGS \
    $TASK_ARGS \
    "$@"