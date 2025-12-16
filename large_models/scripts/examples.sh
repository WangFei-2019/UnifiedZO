# MeZO
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-6.7b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash scripts/mezo.sh

# zero-shot
CUDA_VISIBLE_DEVICES=0 MODEL=quantized/opt-6.7b-gptq-b4-g128 TASK=SST2 bash scripts/icl.sh