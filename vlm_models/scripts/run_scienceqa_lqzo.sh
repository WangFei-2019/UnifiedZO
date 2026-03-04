#!/bin/bash

# 使用 13B 的 GPTQ 模型 
MODEL_NAME="/workspace/wangfei154/models/llava-hf/llava-v1.5-13B-GPTQ" 
OUTPUT_DIR="./checkpoints/scienceqa_lqzo_13b_sanity_check"

echo "=================================================="
echo "启动 VLM 零阶优化 (LQZO) 测试训练"
echo "模型: $MODEL_NAME (13B GPTQ 版本)"
echo "任务: ScienceQA (Image-only Subset)"
echo "=================================================="

# 启动 Python 训练脚本
python run_vlm.py \
    --model_name_or_path $MODEL_NAME \
    --quant_method "gptq" \
    --freeze_vision_tower True \
    --freeze_mm_projector False \
    --output_dir $OUTPUT_DIR \
    --zo_optim "lqzo" \
    --zo_eps 1e-3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 50 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 1 \
    --bf16 True \
    --remove_unused_columns False \
    --report_to "none"