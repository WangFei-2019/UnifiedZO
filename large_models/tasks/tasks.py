from .templates import *
from utils import temp_seed
import json
import os
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Union
import string
import random
import datasets
import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATASET_DIR = "/workspace/wangfei154/datasets/" # None

def get_task(task_name):
    """
    Factory function to retrieve a Dataset instance based on the task name.
    
    Args:
        task_name (str): Name of the task (e.g., "SST2", "GLUE__SST2").
    """
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    
    # Reflective class instantiation: finds the class 'SST2Dataset' for task_group 'SST2'
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    return instance


@dataclass
class Sample:
    """
    Unified representation of a single data sample.
    """
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    """
    Base class for all datasets.
    Handles loading, splitting, and building samples.
    """
    mixed_set = False
    train_sep = "\n\n"
    generation = False # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask
    
    def get_task_name(self):
        return self.subtask
        
    def load_dataset(self):
        """Loads the dataset from Hugging Face Datasets or local path."""
        raise NotImplementedError
    
    def get_template(self, template_version=0):
       """Returns the Template object for prompt formatting."""
       templates = {0: Template}
       return templates[template_version]
   
    def build_sample(self, example):
        """Converts a raw HF example into a Sample object."""
        return 
     
    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        """
        Samples training sets (demonstrations) for Few-Shot Learning or Fine-tuning.
        
        Args:
            num_train (int): Number of training samples (shots).
            num_dev (int): Number of dev samples.
            num_eval (int): Number of evaluation samples.
            num_train_sets (int): How many different sets of demonstrations to sample.
            seed (int): Random seed.
        """
        if seed is not None:
            # One train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # Multiple train/demo sets
            seeds = list(range(num_train_sets))
        else: 
            # One train/demo set per evaluation sample (Standard ICL evaluation)
            assert num_dev is None # dev set not supported in this mode
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = [] 
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:
                raise NotImplementedError("Mixed sets are not fully implemented yet.")
                # train_samples.append(self.sample_subset(data_split="valid", seed=set_seed, num=num_train, exclude=i))
            else:
                if num_dev is not None:
                    # Sample train + dev from the training split
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train+num_dev)) 
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warn("num_train + num_dev > available training examples")
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                
                if num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        """Samples a subset of data with a fixed seed."""
        with temp_seed(seed):
            samples = self.samples[data_split] 
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[:num if exclude is None else num+1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]
    
    @property
    def valid_samples(self):
        return self.samples["valid"]


class SST2Dataset(Dataset):
    """Handler for SST-2 Dataset."""
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        # Loading from GLUE benchmark
        d = load_dataset('glue' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'glue'), 'sst2', trust_remote_code=True)
        train_d = d["train"]
        validation_d = d["validation"]
        
        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        label = int(example["label"])
        # Candidates are implicitly 0 (Negative) and 1 (Positive)
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2Template}[template_version]()
        
    
class CopaDataset(Dataset):
    """Handler for COPA Dataset."""
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        # Loading from SuperGLUE
        train_examples = load_dataset('super_glue' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'super_glue'), "copa", trust_remote_code=True)["train"]
        valid_examples = load_dataset('super_glue' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'super_glue'), "copa", trust_remote_code=True)["validation"]
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


class BoolQDataset(Dataset):
    """Handler for BoolQ Dataset."""
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("boolq" if DATASET_DIR is None else os.path.join(DATASET_DIR, 'boolq'), trust_remote_code=True)
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )
        
        return sample
    
    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


class MultiRCDataset(Dataset):
    """Handler for MultiRC Dataset."""
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset('super_glue' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'super_glue'), "multirc", trust_remote_code=True)
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1], # 0: No, 1: Yes
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):
    """Handler for CB Dataset."""
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset('super_glue' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'super_glue'), "cb", trust_remote_code=True)
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2], # Entailment, Contradiction, Neutral
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):
    """Handler for WiC Dataset."""
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset('super_glue' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'super_glue'), "wic", trust_remote_code=True)
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):
    """Handler for WSC Dataset."""
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        # Note: 'wsc.fixed' is often used in research
        d = load_dataset('super_glue' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'super_glue'), "wsc.fixed",)
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):
    """Handler for ReCoRD Dataset."""
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset('super_glue' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'super_glue'), "record", trust_remote_code=True)
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):
    """Handler for RTE Dataset."""
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset('super_glue' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'super_glue'), "rte", trust_remote_code=True)
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: RTETemplate}[template_version]()

 
class SQuADDataset(Dataset):
    """Handler for SQuAD 2.0 Dataset."""
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("squad" if DATASET_DIR is None else os.path.join(DATASET_DIR, 'squad'), trust_remote_code=True)
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        # Filter examples with no answers if necessary, though SQuAD v1 usually has them
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    """Handler for DROP Dataset."""
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("drop" if DATASET_DIR is None else os.path.join(DATASET_DIR, 'drop'), trust_remote_code=True)
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()


class WinoGrandeDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_set = load_dataset('winogrande' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'winogrande'), 'winogrande_m', split='train', trust_remote_code=True)
        valid_set = load_dataset('winogrande' if DATASET_DIR is None else os.path.join(DATASET_DIR, 'winogrande'), 'winogrande_m', split='validation', trust_remote_code=True)

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = example["sentence"]
        context, target = sentence.split("_")
        sample = Sample(
            data=example,
            candidates=[example['option1'] + target, example['option2'] + target],
            correct_candidate=example[f'option{example["answer"]}'] + target,
        )
        return sample

    def get_template(self, template_version=0):
        if template_version == 0:
            return WinoGrandeTemplate()
        else:
            raise NotImplementedError(f"Template version {template_version} not implemented for WinoGrande")
