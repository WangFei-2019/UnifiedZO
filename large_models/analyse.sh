python layer_importance_analysis.py \
    --model_name /workspace/wangfei154/models/facebook/opt-6.7b \
    --task_name SST2 \
    --output_dir result/analysis \
    --num_train 100 \
    --per_device_train_batch_size 4