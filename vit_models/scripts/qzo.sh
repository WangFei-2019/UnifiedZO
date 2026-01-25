#!/bin/bash

MODEL_NAME="google/vit-base-patch16-224"
TASK_NAME="cifar10"
OUTPUT_DIR="result/vit_qzo_${TASK_NAME}"

# --- Execution ---
# Run QZO training.
# Key Arguments:
#   --quant_method: Specifies the quantization strategy (e.g., 'gptq', 'aqlm', or 'none' for simulated).
#   --zo_scale: Learning rate scaler for the quantization step size parameters.
#   --train_unquantized: Whether to train parameters that are kept in FP32 (e.g., LayerNorm).
python vit_models/run_vit.py \
    --model_name $MODEL_NAME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train 1000 \
    --num_eval 100 \
    --trainer qzo \
    --learning_rate 1e-5 \
    --zo_eps 1e-3 \
    --zo_scale 1.0 \
    --quant_method none \
    --train_unquantized True \
    --per_device_train_batch_size 16 \
    --train_as_classification True \
    --report_to wandb \
    --tag "qzo_benchmark"