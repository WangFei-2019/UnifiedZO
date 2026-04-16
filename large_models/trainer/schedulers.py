import math
import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)

# --- Learning Rate Schedulers ---

def get_constant_schedule(learning_rate, current_step, num_training_steps, **kwargs):
    return learning_rate

def get_constant_with_warmup_schedule(learning_rate, current_step, num_training_steps, num_warmup_steps=0, **kwargs):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps)) * learning_rate
    return learning_rate

def get_linear_schedule_with_warmup(learning_rate, current_step, num_training_steps, num_warmup_steps=0, **kwargs):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * learning_rate
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))) * learning_rate

def get_cosine_schedule_with_warmup(learning_rate, current_step, num_training_steps, num_warmup_steps=0, **kwargs):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * learning_rate
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * progress))) * learning_rate

# --- Hessian Smoothing Schedulers (for HiZOO) ---

def get_constant_smoothing(current_step, num_training_steps, value=1e-8):
    return value

def get_linear_decay_smoothing(current_step, num_training_steps, start_value=1e-4, end_value=1e-10):
    progress = current_step / num_training_steps
    return start_value + (end_value - start_value) * progress

# --- Dispatchers ---

LR_SCHEDULER_MAP = {
    'constant': get_constant_schedule,
    'constant_with_warmup': get_constant_with_warmup_schedule,
    'linear_with_warmup': get_linear_schedule_with_warmup,
    'cosine_with_warmup': get_cosine_schedule_with_warmup,
}

HESSIAN_SCHEDULER_MAP = {
    'constant0': lambda c, t: 0.0,
    'constant1e-6': lambda c, t: 1e-6,
    'constant1e-8': lambda c, t: 1e-8,
    'constant1e-10': lambda c, t: 1e-10,
    'linear_decay': get_linear_decay_smoothing
}

def zo_lr_scheduler(learning_rate, scheduler_type, current_step, num_training_steps, num_warmup_steps=0, num_decay_steps=0):
    """
    Retrieves the learning rate based on the current step and configuration.
    """
    if scheduler_type not in LR_SCHEDULER_MAP:
        # Default to constant if unknown
        return learning_rate
    
    return LR_SCHEDULER_MAP[scheduler_type](
        learning_rate, 
        current_step, 
        num_training_steps, 
        num_warmup_steps=num_warmup_steps
    )

def hessian_smooth_scheduler(scheduler_type, current_step, num_training_steps):
    """
    Retrieves the smoothing factor for HiZOO Hessian estimation.
    """
    if scheduler_type not in HESSIAN_SCHEDULER_MAP:
        return 1e-8
    return HESSIAN_SCHEDULER_MAP[scheduler_type](current_step, num_training_steps)

def _get_learning_rate(self):
    return zo_lr_scheduler(
        learning_rate=self.args.learning_rate,
        scheduler_type=getattr(self.args, 'zo_lr_scheduler_type', 'constant_with_warmup'),
        current_step=self.state.global_step,
        num_training_steps=self.state.max_steps,
        num_warmup_steps=getattr(self.args, 'warmup_steps', 0),
        num_decay_steps=getattr(self.args, 'decay_step', 0) 
    )

