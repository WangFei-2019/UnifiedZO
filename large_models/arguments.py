from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class ZOTrainingArguments(TrainingArguments):
    """
    Arguments for Zeroth-Order Optimization training.
    Inherits from transformers.TrainingArguments to support standard HF features.
    """

    # --- Task and Model Arguments ---
    task_name: str = field(
        default="SST2", 
        metadata={"help": "Task name (e.g., SST2, BoolQ, SQuAD). Must match class names in tasks/tasks.py"}
    )
    model_name: str = field(
        default="facebook/opt-125m",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    load_float16: bool = field(default=False, metadata={"help": "Load model in fp16."})
    load_bfloat16: bool = field(default=False, metadata={"help": "Load model in bf16."})
    load_int8: bool = field(default=False, metadata={"help": "Load model in int8 (8-bit quantization)."})
    no_auto_device: bool = field(default=False, metadata={"help": "Disable auto device mapping (useful for FSDP)."})
    max_length: int = field(default=2048, metadata={"help": "Max sequence length."})

    # --- Scheduler & Optimizer Arguments ---
    lr_scheduler_type: str = field(default="constant", metadata={"help": "ZO learning rate scheduler type."})
    
    # --- Dataset / Sampling Arguments ---
    num_train: int = field(default=0, metadata={"help": "Number of training samples."})
    num_dev: int = field(default=None, metadata={"help": "Number of development samples."})
    num_eval: int = field(default=None, metadata={"help": "Number of evaluation samples."})
    train_set_seed: int = field(default=None, metadata={"help": "Seed for sampling training set."})
    num_train_sets: int = field(default=None, metadata={"help": "Number of different training sets to sample."})


    # --- Calibration (SFC) ---
    sfc: bool = field(default=False, metadata={"help": "Surface Form Competition calibration."})
    icl_sfc: bool = field(default=False, metadata={"help": "SFC for In-Context Learning"})

    template_ver: int = field(default=0, metadata={"help": "template. For some tasks (SST2, RTE, Copa), we add template ver=1 as the empty template."})

    # --- ZO Specific Arguments ---
    trainer: str = field(
        default="none", 
        metadata={"help": "The ZO method to use: 'mezo', 'zoadamu', 'lozo', 'hizoo', 'pzo' (pseuZO), 'fzoo'. The FO method to use: 'regular'. ICL method to use: 'none'"}
    )
    zo_eps: float = field(default=1e-3, metadata={"help": "Epsilon value for perturbation scale."})

    # --- ZO-AdaMU Specific Arguments ---
    zo_adamu_beta1: float = field(default=0.9, metadata={"help": "Beta1 parameter for ZO-AdaMU (momentum decay)."})
    zo_adamu_beta2: float = field(default=0.999, metadata={"help": "Beta2 parameter for ZO-AdaMU (variance decay)."})
    zo_adamu_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon parameter for ZO-AdaMU stability."})

    # --- LoZO (Low-Rank ZO) Arguments ---
    lozo_rank: int = field(default=2, metadata={"help": "Rank (r) for LoZO matrix decomposition."})
    lozo_step_interval: int = field(default=50, metadata={"help": "Step interval to update the projection matrix V in LoZO."})
    
    # --- HiZOO (Hessian-guided ZO) Arguments ---
    hessian_smooth: float = field(default=1e-8, metadata={"help": "Smoothing factor for Hessian estimation."})
    hessian_smooth_type: str = field(default="constant0", metadata={"help": "Scheduler type for Hessian smoothing."})
    
    # --- PseuZO Arguments ---
    sliding_window_length: int = field(default=14, metadata={"help": "Length of the sliding window for gradient averaging."})
    momentum_fb_min: float = field(default=0.0, metadata={"help": "Minimum momentum factor."})
    momentum_fb_max: float = field(default=1.0, metadata={"help": "Maximum momentum factor."})
    perturb_type: str = field(default="Gaussian", metadata={"help": "Type of random noise: Gaussian, Rademacher, etc."})
    logits: bool = field(default=False, metadata={"help": "If True, use logits for PseuZO projection. If False, use last hidden state."})
    adjust_lm_head: bool = field(default=False, metadata={"help": "Whether to adjust LM head gradients in PZO."})
    hmezo_l: bool = field(default=True, metadata={"help": "HMeZO specific flag (if using HMeZO logic)."})

    # --- FZOO Specific Arguments ---
    fzoo_n: int = field(default=8, metadata={"help": "Number of perturbations (N) for FZOO."})
    fzoo_thre: int = field(default=0, metadata={"help": "Threshold step for FZOO logic."})
    fzoo_d: int = field(default=1, metadata={"help": "D parameter for FZOO (Strategy selection)."})

    # --- AdaLeZO Specific Arguments ---
    adalezo_k_ratio: float = field(
        default=0.1, 
        metadata={"help": "Ratio of layers to sample per step (e.g., 0.1 means 10% of layers are active)."}
    )
    adalezo_tau: float = field(
        default=0.1, 
        metadata={"help": "Temperature for Softmax exploration in layer selection."}
    )
    adalezo_c: float = field(
        default=0.7, 
        metadata={"help": "UCB Exploration Constant. Larger values encourage exploring less selected layers."}
    )
    adalezo_ipw_clip: float = field(
        default=10.0, 
        metadata={"help": "Maximum clipping value for Inverse Probability Weighting (IPW) to prevent gradient explosion."}
    )
    adalezo_layer_momentum: bool = field(
        default=False, 
        metadata={"help": "Enable layer-wise adaptive scaling (RMSProp-style variance tracking)."}
    )
    adalezo_beta: float = field(
        default=0.95, 
        metadata={"help": "Decay factor for the layer-wise adaptive scaling."}
    )
    adalezo_warm_start: bool = field(
        default=False, 
        metadata={"help": "Warm-start probabilities based on layer depth (bias towards deeper layers initially)."}
    )
    adalezo_gamma: float = field(
        default=0.01, 
        metadata={"help": "Mixing factor for uniform distribution to ensure non-zero probability."}
    )
    adalezo_interval: int = field(
        default=1, 
        metadata={"help": "Re-sample active layers every N steps (Stickiness strategy)."}
    )

    # --- Training Strategy ---
    only_train_option: bool = field(default=True, metadata={"help": "If True, only calculates loss on the answer/option part."})
    train_as_classification: bool = field(default=False, metadata={"help": "If True, computes log-likelihood of all options and trains as classification."})
    non_diff: bool = field(default=False, metadata={"help": "Use non-differentiable objective (e.g., F1 score for SQuAD)."})
    
    # --- PEFT (LoRA / Prefix) ---
    prefix_tuning: bool = field(default=False, metadata={"help": "Use Prefix Tuning."})
    num_prefix: int = field(default=5, metadata={"help": "Number of prefix tokens."})
    reparam: bool = field(default=False, metadata={"help": "Reparameterization for prefix tuning."})
    prefix_init_by_real_act: bool = field(default=True, metadata={"help": "Initialize prefix by real activations."})
    
    lora: bool = field(default=False, metadata={"help": "Use LoRA."})
    lora_r: int = field(default=8, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})

    # --- Prompt Tuning Arguments ---
    prompt_tuning: bool = field(
        default=False, 
        metadata={"help": "Whether to use Prompt Tuning."}
    )
    num_virtual_tokens: int = field(
        default=10, 
        metadata={"help": "Number of virtual prompt tokens to prepend."}
    )
    prompt_init_by_real_tokens: bool = field(
        default=False, 
        metadata={"help": "Whether to initialize prompt tokens using random real word embeddings from the model vocabulary."}
    )

    # --- Generation Args ---
    sampling: bool = field(default=False, metadata={"help": "Use sampling for generation."})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for generation."})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for generation."})
    top_k: Optional[int] = field(default=None, metadata={"help": "Top-k for generation."})
    top_p: float = field(default=0.95, metadata={"help": "Top-p for generation."})
    max_new_tokens: int = field(default=50, metadata={"help": "Max new tokens to generate."})
    eos_token: str = field(default="\n", metadata={"help": "End of sentence token."})

    # --- Misc ---
    save_model: bool = field(default=False, metadata={"help": "Whether to save the final model."})
    save_on_interrupt: bool = field(default=False, metadata={"help": "Save model on SIGUSR1/SIGINT interrupt."})
    verbose: bool = field(default=False, metadata={"help": "Enable verbose logging."})
    tag: str = field(default="", metadata={"help": "Tag for result file naming."})
    result_file: str = field(default=None, metadata={"help": "Specific path to save results."})
    no_eval: bool = field(default=False, metadata={"help": "Disable evaluation during training."})

    # Linear Probing options
    linear_probing: bool = field(default=False, metadata={"help": "Use linear probing."})
    lp_early_stopping: bool = field(default=False, metadata={"help": "Early stopping for linear probing."})

    head_tuning: bool = field(default=False, metadata={"help": "Tune only the LM head."})
    untie_emb: bool = field(default=False, metadata={"help": "Untie embeddings and LM head."})

    report_to: str = field(default="wandb", metadata={"help": "Where to report results (e.g., 'wandb', 'tensorboard', 'none')."})
