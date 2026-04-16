# AdaLeZO: Universally Empowering Zeroth-Order Optimization via Adaptive Layer-wise Sampling

Official implementation of the paper: **"Universally Empowering Zeroth-Order Optimization via Adaptive Layer-wise Sampling"** (ACL 2026 Findings).

AdaLeZO is an adaptive Zeroth-Order (ZO) optimization framework designed for memory-efficient fine-tuning of Large Language Models (LLMs). By formulating layer selection as a non-stationary Multi-Armed Bandit (MAB) problem, AdaLeZO dynamically identifies and updates the most sensitive layers, achieving **1.7x to 3.0x wall-clock acceleration** compared to state-of-the-art ZO methods without additional memory overhead.


## Overview

This repository contains a unified implementation of zeroth-order optimization methods across:

- Large language models (`large_models/`)
- Vision transformers (`vit_models/`)
- Vision-language models (`vlm_models/`)
- Shared zeroth-order training core (`zo_core/`)

## Repository Structure

```text
UnifiedZO/
├── large_models/
├── vit_models/
├── vlm_models/
└── zo_core/
```

## Environment Setup

### 1) Create environment

```bash
conda create -n unifiedzo python=3.12 -y
conda activate unifiedzo
```

### 2) Install core dependencies

```bash
pip install torch==2.10.0 torchvision==0.25.0 triton==3.6.0
pip install transformers==4.57.3 peft==0.18.0 accelerate==1.12.0 datasets==3.6.0 tokenizers==0.22.2 wandb==0.23.1
pip install numpy==2.2.6 pandas==2.3.3 scipy==1.16.3 scikit-learn==1.8.0 tqdm==4.67.1
```

### 3) Reproducibility (optional)

```bash
pip freeze > requirements.txt
# Optional: avoid online logging in first run
export WANDB_MODE=offline
```

## Quick Start

### Large Models

```bash
cd large_models
bash scripts/adalezo.sh
```

### Vision Models

```bash
cd vit_models
bash scripts/mezo.sh
```

### Vision-Language Models

```bash
cd vlm_models
bash scripts/run_scienceqa_lqzo.sh
```

## Notes

- Please adjust dataset paths, model checkpoints, and runtime arguments in scripts before launching experiments.
- Method-specific training logic is implemented in `zo_core/trainer/`.

## Citation

```bibtex
@inproceedings{wang2026adalezo,
  title={Universally Empowering Zeroth-Order Optimization via Adaptive Layer-wise Sampling},
  author={Wang, Fei and Shen, Li and Ding, Liang and Xue, Chao and Liu, Ye and Ding, Changxing},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```