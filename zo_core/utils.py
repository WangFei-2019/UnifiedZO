import logging
import time
import contextlib
import numpy as np
import json
import os
import torch
import random
from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Dict, Any, Union, Optional
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Helper to set seed for reproducibility.
    Similar to transformers.set_seed but valid for core usage.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True 
    # torch.backends.cudnn.benchmark = False

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Extended JSON Encoder to handle Dataclasses and other types.
    """
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        return super().default(o)

def write_metrics_to_file(metrics, output):
    """
    Writes a dictionary of metrics to a JSON file.
    """
    if not output:
        return
    # Ensure directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    with open(output, "w") as f:
        json.dump(metrics, f, cls=EnhancedJSONEncoder, indent=4)

@contextlib.contextmanager
def count_time(name):
    """
    Context manager to log the duration of a block of code.
    """
    logger.info(f"{name}...")
    start_time = time.time()
    try:
        yield
    finally:
        logger.info(f"Done with {name}: {time.time() - start_time:.2f}s")

@contextlib.contextmanager
def temp_seed(seed):
    """
    Temporarily set the numpy random seed, then restore the state.
    Useful for deterministic sampling in a randomized environment.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class HFDataset(Dataset):
    """
    Simple wrapper to convert a list of dictionaries into a PyTorch Dataset.
    Compatible with Hugging Face Trainer and standard PyTorch Dataloaders.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def result_file_tag(args):
    """
    Generates a unique tag for the result file based on arguments.
    Modified to be robust (uses getattr) so it works for both NLP and Vision args.
    """
    # Fallback if model_name is a path
    model_name = args.model_name.split("/")[-1] if hasattr(args, 'model_name') else "model"
    
    # NLP specific tags (optional, only if they exist)
    sfc_tag = "-sfc" if (getattr(args, 'sfc', False)) else ""
    icl_sfc_tag = "-icl_sfc" if (getattr(args, 'icl_sfc', False)) else ""
    
    # Common tags
    # num_eval / num_train might be None or 0
    num_eval = getattr(args, 'num_eval', None)
    sample_eval_tag = "-sampleeval%d" % num_eval if num_eval is not None else ""
    
    num_train = getattr(args, 'num_train', 0)
    sample_train_tag = "-ntrain%d" % num_train if num_train > 0 else ""
    
    num_dev = getattr(args, 'num_dev', None)
    sample_dev_tag = "-ndev%d" % num_dev if num_dev is not None else ""
    
    # Custom tag
    tag_val = getattr(args, 'tag', "")
    customized_tag = f"-{tag_val}" if tag_val else ""
    
    task_name = getattr(args, 'task_name', 'task')
    
    return f"{task_name}-{model_name}{sfc_tag}{icl_sfc_tag}{sample_eval_tag}{sample_train_tag}{sample_dev_tag}{customized_tag}"