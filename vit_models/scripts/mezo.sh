#!/bin/bash
# Define the target model and task
# We use the standard ViT-base model and CIFAR-10 dataset as the default benchmark.
MODEL_NAME="google/vit-base-patch16-224"
TASK_NAME="cifar10"
OUTPUT_DIR="result/vit_mezo_${TASK_NAME}"

# --- Execution ---
# Run the ViT training entry point with MeZO trainer.
# Note: 'zo_eps' controls the perturbation scale, 'zo_sample' is usually 1 for MeZO.
python vit_models/run_vit.py \
    --model_name $MODEL_NAME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train 1000 \
    --num_eval 100 \
    --trainer mezo \
    --learning_rate 1e-5 \
    --zo_eps 1e-3 \
    --max_length 224 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --train_as_classification True \
    --save_model False \
    --report_to wandb \
    --tag "mezo_benchmark"