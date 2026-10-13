# UnifiedZO

UnifiedZO is the shared codebase for **AdaLeZO** and **LQZO**, providing zeroth-order (ZO) fine-tuning infrastructure for large language models and vision transformers, together with additional vision-language experiments.

## Papers

| Method | Paper | Venue |
| --- | --- | --- |
| **AdaLeZO** | [Universally Empowering Zeroth-Order Optimization via Adaptive Layer-wise Sampling](https://aclanthology.org/2026.findings-acl.1999/) | Findings of ACL 2026 |
| **LQZO** | [Improving Quantized Zeroth-Order Optimization through Reconstructed Low-Rank Structures](https://openreview.net/forum?id=1UKJD5PlWg) | NeurIPS 2026 |

### AdaLeZO: adaptive layer-wise sampling

AdaLeZO allocates the perturbation budget across layers by modeling layer selection as a non-stationary multi-armed bandit problem. It uses inverse probability weighting to account for non-uniform sampling and can be applied to several ZO optimizers. The paper reports **1.7x–3.0x wall-clock acceleration** in its evaluated settings.

- Training implementation: [`zo_core/trainer/adalezo_trainer.py`](zo_core/trainer/adalezo_trainer.py)
- LLM launcher: [`large_models/scripts/adalezo.sh`](large_models/scripts/adalezo.sh)
- Adaptive variants of other ZO methods: `large_models/scripts/ada*.sh`

### LQZO: low-rank quantized zeroth-order optimization

LQZO improves adaptation of quantized models by reconstructing low-rank perturbations for quantization-scale optimization. The paper combines low-rank factorization, Binary-Aggregated Gaussian Perturbation, and Isotropy-Enhanced Estimation. Evaluations cover LLMs at **INT8, INT4, and INT2**, as well as RTN-quantized vision transformers.

- Training implementation: [`zo_core/trainer/lqzo_trainer.py`](zo_core/trainer/lqzo_trainer.py)
- LLM launcher: [`large_models/scripts/lqzo.sh`](large_models/scripts/lqzo.sh)
- ViT launcher: [`vit_models/scripts/lqzo.sh`](vit_models/scripts/lqzo.sh)

AdaLeZO focuses on adaptive layer selection, while LQZO focuses on structured updates to quantization scales. Both use the shared training infrastructure in `zo_core/`.

## Repository Structure

```text
UnifiedZO/
├── large_models/        # LLM fine-tuning, evaluation, and GPTQ quantization
│   └── scripts/        # AdaLeZO, LQZO, and baseline launchers
├── vit_models/          # Vision-transformer experiments
│   └── scripts/        # LQZO, QZO, MeZO, and other launchers
├── vlm_models/          # Additional vision-language experiments
└── zo_core/
    ├── arguments.py    # Shared training arguments
    ├── trainer/        # ZO optimizers and adaptive variants
    └── tuners/         # Parameter-efficient tuning modules
```

## Environment Setup

### 1. Clone the repository and create an environment

```bash
git clone https://github.com/WangFei-2019/UnifiedZO.git
cd UnifiedZO
conda create -n unifiedzo python=3.12 -y
conda activate unifiedzo
```

### 2. Install core dependencies

```bash
pip install torch==2.10.0 torchvision==0.25.0 triton==3.6.0
pip install transformers==4.57.3 peft==0.18.0 accelerate==1.12.0 datasets==3.6.0 tokenizers==0.22.2 wandb==0.23.1
pip install numpy==2.2.6 pandas==2.3.3 scipy==1.16.3 scikit-learn==1.8.0 tqdm==4.67.1
```

GPTQ workflows additionally require `gptqmodel`. Install a build compatible with your PyTorch and CUDA environment.

### 3. Configure models and datasets

- For LLM tasks, configure `DATASET_DIR` in [`large_models/tasks/tasks.py`](large_models/tasks/tasks.py). Set it to your local dataset root, or to `None` to load datasets from Hugging Face.
- For GPTQ-based LQZO runs, set `MODEL` to a prepared GPTQ checkpoint. [`large_models/quantization.py`](large_models/quantization.py) provides a quantization utility; configure its calibration dataset path before use.
- Launcher defaults are examples. Use the model, task, bit width, learning rate, and other settings from the corresponding paper for paper-specific experiments.
- Optional offline experiment logging:

```bash
export WANDB_MODE=offline
```

## Quick Start

Run each example from the **repository root**, after configuring model and dataset paths.

### AdaLeZO on an LLM

```bash
(
  cd large_models
  TASK=SST2 MODEL=facebook/opt-6.7b bash scripts/adalezo.sh
)
```

Important layer-selection arguments include `--adalezo_k_ratio`, `--adalezo_tau`, `--adalezo_ema_alpha`, and `--adalezo_ipw_clip`. See [`zo_core/arguments.py`](zo_core/arguments.py) for their definitions.

### LQZO on a GPTQ-quantized LLM

Replace the checkpoint path below with your prepared model:

```bash
(
  cd large_models
  TASK=SST2 MODEL=/path/to/opt-6.7b-gptq-b4-g128 \
    LOZO_RANK=2 LOZO_INTERVAL=50 \
    bash scripts/lqzo.sh --channel_scale 8 --train_unquantized False
)
```

| Paper parameter | Training argument | Example value |
| --- | --- | --- |
| Perturbation rank `r` | `--lozo_rank` | `2` |
| Lazy update interval `tau` | `--lozo_step_interval` | `50` |
| Reshaping factor `gamma` | `--channel_scale` | `8` |
| Scalar gradient clipping | `--clip_zo_grad` | Enabled by the LLM launcher |

### LQZO on a vision transformer

The ViT launcher invokes `vit_models/run_vit.py`, so run it from the repository root:

```bash
MODEL=google/vit-base-patch16-224 TASK=uoft-cs/cifar10 \
  BIT=4 LOZO_RANK=2 CHANNEL_SCALE=8 INTERVAL=50 \
  bash vit_models/scripts/lqzo.sh
```

The repository also includes QZO and MeZO baselines. Additional VLM experimental code is available in `vlm_models/`; the LQZO paper reports LLM and ViT evaluations.

## Citation

Please cite the paper corresponding to the method you use. Cite both papers when using both methods.

### AdaLeZO

```bibtex
@inproceedings{wang2026adalezo,
  title={Universally Empowering Zeroth-Order Optimization via Adaptive Layer-wise Sampling},
  author={Wang, Fei and Shen, Li and Ding, Liang and Xue, Chao and Liu, Ye and Ding, Changxing},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026},
  publisher={Association for Computational Linguistics},
  pages={40218--40239},
  doi={10.18653/v1/2026.findings-acl.1999},
  url={https://aclanthology.org/2026.findings-acl.1999/}
}
```

### LQZO

```bibtex
@inproceedings{wang2026lqzo,
  title={Improving Quantized Zeroth-Order Optimization through Reconstructed Low-Rank Structures},
  author={Wang, Fei and Xie, Shuai and Shen, Li and Xue, Chao and Liu, Ye and Ding, Changxing},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=1UKJD5PlWg}
}
```

## Contact

For questions about this repository, contact **Fei Wang** at [wang_fei2019@163.com](mailto:wang_fei2019@163.com).
