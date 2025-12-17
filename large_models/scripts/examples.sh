# # Zero-shot
CUDA_VISIBLE_DEVICES=0 MODEL=quantized/opt-6.7b-gptq-b4-g128 TASK=SST2 bash scripts/icl.sh

# In-context learning 32-shot
CUDA_VISIBLE_DEVICES=0 MODEL=quantized/opt-6.7b-gptq-b4-g128 TASK=SST2 NUM_TRAIN=32 bash scripts/icl.sh

# Full-parameter fine-tuning
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-5 bash finetune.sh

# LoRA
MODEL=facebook/opt-1.3b TASK=SST2 MODE=lora LR=1e-4 bash finetune.sh

# prefix-tuning
MODEL=facebook/opt-1.3b TASK=SST2 MODE=prefix LR=1e-2 bash finetune.sh

# MeZO
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-6.7b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash scripts/mezo.sh