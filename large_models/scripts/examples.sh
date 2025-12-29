# # Zero-shot
CUDA_VISIBLE_DEVICES=0 MODEL=quantized/opt-6.7b-gptq-b4-g128 TASK=SST2 bash scripts/icl.sh

# In-context learning 32-shot
CUDA_VISIBLE_DEVICES=0 MODEL=quantized/opt-6.7b-gptq-b4-g128 TASK=SST2 NUM_TRAIN=32 bash scripts/icl.sh

# Full-parameter fine-tuning
MODEL=meta/opt-1.3b TASK=SST2 MODE=ft LR=1e-5 bash scripts/finetune.sh
# for OPT series
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-5 bash scripts/finetune.sh --load_float32

# LoRA
MODEL=facebook/opt-1.3b TASK=SST2 MODE=lora LR=1e-4 bash scripts/finetune.sh

# prefix-tuning
MODEL=facebook/opt-1.3b TASK=SST2 MODE=prefix LR=1e-2 bash scripts/finetune.sh

# MeZO
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-6.7b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash scripts/mezo.sh


# run for Qwen-2.5-1.5b
CUDA_VISIBLE_DEVICES=0 MODEL=/workspace/wangfei154/models/Qwen/Qwen2.5-1.5B TASK=SST2 LR=1e-6 EPS=1e-5 STEPS=5000 EVAL_STEPS=500 bash scripts/mezo.sh 