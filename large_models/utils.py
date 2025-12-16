import logging
import time
import contextlib
import numpy as np
from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Dict, Any, Union, Optional
import json

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

logger = logging.getLogger(__name__)

def encode_prompt(task, template, train_samples, eval_sample, tokenizer, max_length, 
                  sfc=False, icl_sfc=False, generation=False, generation_with_gold=False, max_new_tokens=None):
    """
    Encode prompts for eval_sample
    Input: 
    - task, template: task and template class
    - train_samples, eval_sample: demonstrations and the actual sample
    - tokenizer, max_length: tokenizer and max length
    - sfc: generate prompts for calibration (surface form competition; https://arxiv.org/abs/2104.08315)
    - icl_sfc: generate prompts for ICL version calibration
    - generation: whether it is an generation task
    - generation_with_gold: whether to include the generation-task gold answers (for training)
    - max_new_tokens: max number of new tokens to generate so that we can save enough space 
      (only for generation tasks)
    Output:
    - encodings: a list of N lists of tokens. N is the number of options for classification/multiple-choice.
    - option_lens: a list of N integers indicating the number of option tokens.
    """

    # Demonstrations for ICL
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    train_prompts = task.train_sep.join(train_prompts).strip()
    
    # sfc or icl_sfc indicates that this example is used for calibration
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc
        verbalize_fn = template.verbalize_sfc
    else: 
        encode_fn = template.encode
        verbalize_fn = template.verbalize 
            
    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')
    
    if not generation:
        # We generate one prompt for each candidate (different classes in classification)
        # or different choices in multiple-choice tasks
        verbalized_eval_prompts = [verbalize_fn(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
        
        # Calculate option length by subtracting prompt length from full length
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]

        if sfc:
            # SFC doesn't use context/demonstrations
            final_prompts = verbalized_eval_prompts 
        else:
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
    else:
        # For Generation: Just prompt
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            # Training mode: include answer
            verbalized_eval_prompts = [verbalize_fn(eval_sample, eval_sample.correct_candidate)]
            unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
            option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
        else:
            # Inference mode: input only
            option_lens = [0]
            final_prompts = [(train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(' ')]

    # Tokenize
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    # Truncate (left truncate as demonstrations are less important)
    if generation and max_new_tokens is not None:
        max_length = max_length - max_new_tokens

    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warning(f"Exceed max length {max_length}, truncating...")
        
    # Apply truncation logic (keep the end)
    if getattr(tokenizer, "add_bos_token", False):
        # Keep BOS at start, truncate middle/left
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]  
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]  
   
    return encodings, option_lens

# --- Data Collators ---

@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator that handles nested lists (common in multiple-choice tasks) and pads.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Flatten for padding
        flattened_features = [ff for f in features for ff in f]
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Standardize label key
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
            
        return batch

@dataclass
class NondiffCollator:
    """
    Collator for non-differentiable objectives (passes 'gold' answer text).
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        import torch
        
        # Extract gold if present
        gold = [f.pop("gold") for f in features] if "gold" in features[0] else None
        
        # Use standard padding for input_ids
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Handle labels for padding
        if "input_ids" in batch:
            # Create labels matching input_ids but with masking logic if needed
            # For simplicity in ZO, we often rely on the model wrapper to handle masking via option_len
            # Here we just pass inputs as labels or clone them
            batch["labels"] = batch["input_ids"].clone()

        if gold is not None:
            batch["gold"] = gold
        
        return batch

# --- Helpers ---

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)

def write_metrics_to_file(metrics, output):
    with open(output, "w") as f:
        json.dump(metrics, f, cls=EnhancedJSONEncoder, indent=4)

@contextlib.contextmanager
def count_time(name):
    print(f"{name}...")
    start_time = time.time()
    try:
        yield
    finally:
        print(f"Done with {name}: {time.time() - start_time:.2f}s")

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class HFDataset(Dataset):
    """
    Simple wrapper to convert a list of dictionaries into a PyTorch Dataset 
    compatible with Hugging Face Trainer.
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
    Useful for organizing experiment results.
    """
    save_model_name = args.model_name.split("/")[-1]
    
    # Optional tags based on flags
    sfc_tag = "-sfc" if (hasattr(args, 'sfc') and args.sfc) else ""
    icl_sfc_tag = "-icl_sfc" if (hasattr(args, 'icl_sfc') and args.icl_sfc) else ""
    
    # Sample counts
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    
    # Custom tag
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    
    return f"{args.task_name}-{save_model_name}{sfc_tag}{icl_sfc_tag}{sample_eval_tag}{sample_train_tag}{sample_dev_tag}{customized_tag}"


def process_dataset(
    args, 
    task, 
    samples, 
    tokenizer, 
    is_training=True
):
    """
    Process raw data samples into tokenized features ready for the model.
    Handles template encoding, label creation, and formatting for different training modes.
    """
    data = []
    
    # Check if we should perform SFC (Surface Form Competition) calibration
    sfc = getattr(args, 'sfc', False)
    icl_sfc = getattr(args, 'icl_sfc', False)

    for sample in samples:
        # Encode the prompt using the task's template
        # returns: encoded_candidates (list of token ids), option_lens (list of int)
        encoded_candidates, option_lens = encode_prompt(
            task, 
            task.get_template(), 
            [], # Train samples (demonstrations) - empty for standard fine-tuning usually
            sample, 
            tokenizer, 
            max_length=args.max_length, 
            sfc=sfc, 
            icl_sfc=icl_sfc,
            generation=task.generation, 
            generation_with_gold=is_training, # Include answer if training
            max_new_tokens=args.max_new_tokens
        )

        # Determine the correct candidate ID (Label)
        if task.generation:
            correct_candidate_id = 0
        elif isinstance(sample.correct_candidate, list):
            correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
        else:
            correct_candidate_id = sample.candidates.index(sample.correct_candidate)
        
        # Logic for Non-Differentiable Objectives (e.g., SQuAD F1)
        if args.non_diff:
            # Remove the answer part from input because we want the model to generate it
            encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

        # --- Format Data for Trainer ---
        
        if args.train_as_classification:
            # Case 1: Classification (Rank all options)
            # We create a list of inputs, one for each option.
            # The collator will handle nesting.
            item = []
            for _i in range(len(encoded_candidates)):
                item.append({
                    "input_ids": encoded_candidates[_i],
                    "labels": correct_candidate_id, # Label is the index of the correct option
                    "option_len": option_lens[_i],
                    "num_options": len(sample.candidates)
                })
            data.append(item)
            
        elif args.only_train_option:
            # Case 2: Causal LM on Answer Only (Standard MeZO/ZO)
            # We only train on the correct candidate. 
            # The custom forward pass will use 'option_len' to mask the prompt loss.
            item = {
                "input_ids": encoded_candidates[correct_candidate_id],
                "labels": encoded_candidates[correct_candidate_id], # Labels = Input (Autoregressive)
                "option_len": option_lens[correct_candidate_id]
            }
            if args.non_diff:
                item["gold"] = sample.correct_candidate
            data.append(item)
            
        else:
            # Case 3: Standard Causal LM (Train on whole sequence)
            data.append({
                "input_ids": encoded_candidates[correct_candidate_id],
                "labels": encoded_candidates[correct_candidate_id]
            })
        
    return HFDataset(data)