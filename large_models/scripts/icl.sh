MODEL=${MODEL:-facebook/opt-13b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

# zero_shot icl evaluation
python run.py --model_name $MODEL --task_name $TASK --output_dir result-tmp --tag icl --num_train 0 --num_eval 1000 --trainer none --seed 0 --output_dir result_zero-shot/$TASK-${MODEL_NAME} "$@" # --verbose

# few_shot (32) icl evaluation
# python run.py --model_name $MODEL --task_name $TASK --output_dir result-tmp --tag icl --num_train 32 --num_eval 1000 --trainer none --seed 0 --output_dir result_icl-32/$TASK-${MODEL_NAME} "$@" # --verbose