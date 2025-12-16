MODEL=${MODEL:-facebook/opt-13b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
NUM_TRAIN=${NUM_TRAIN:-0} # 32

# zero_shot icl or few_shot (32) evaluation
python run.py --model_name $MODEL --task_name $TASK --output_dir result-tmp --tag icl --num_train $NUM_TRAIN --num_eval 1000 --trainer none --seed 0 --output_dir result-none/$TASK-${MODEL_NAME}-${NUM_TRAIN} "$@" # --verbose