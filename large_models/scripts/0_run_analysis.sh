#!/bin/bash
# 运行分析脚本，获取 "Oracle" 梯度分布
export CUDA_VISIBLE_DEVICES="0,1"

# 使用较小的模型进行快速验证 (如 OPT-1.3B)
MODEL=/workspace/wangfei154/models/facebook/opt-6.7b
TASK=SST2

python get_gradient_stats.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir analysis_results \
    --num_train 1000 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --train_as_classification \
    --seed 0