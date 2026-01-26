import logging
import os
import glob
from datasets import load_dataset, load_from_disk
from typing import Dict, Any, Optional

# Initialize logger for the module
logger = logging.getLogger(__name__)

# --- Configuration ---
# Allow dynamic configuration via environment variables for flexibility across different compute environments.
# Defaults to the specific user path if the environment variable is not set.
DATASET_DIR = os.getenv("DATASET_DIR", "/workspace/wangfei154/datasets/")

class VisionDataset:
    """
    Base class for vision datasets, managing data loading, split assignment, and metadata extraction.
    
    This class implements a multi-strategy loading mechanism to ensure robustness across 
    different storage formats (Arrow, Parquet, Python scripts) and environments (Local vs. Online).
    """
    def __init__(self, dataset_name: str, config_name: Optional[str] = None):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.num_labels = 0
        self.id2label = {}
        self.label2id = {}
        self.train_dataset = None
        self.eval_dataset = None
        
        self._load_dataset()

    def _load_dataset(self):
        """
        Executes the dataset loading pipeline with the following priority:
        1. Local 'save_to_disk' format (Arrow).
        2. Local Parquet files (automatically detected).
        3. Local standard dataset script (if available).
        4. Remote Hugging Face Hub (Online fallback).
        """
        logger.info(f"Initiating dataset loading for: {self.dataset_name}")
        
        # 1. Path Construction
        # Construct potential local file paths based on the dataset name and the root directory.
        possible_paths = []
        if DATASET_DIR:
            # Case A: Full relative path (e.g., .../datasets/uoft-cs/cifar-10)
            possible_paths.append(os.path.join(DATASET_DIR, self.dataset_name))
            # Case B: Suffix only (e.g., .../datasets/cifar-10) - handles cases where user ignored the namespace.
            if "/" in self.dataset_name:
                possible_paths.append(os.path.join(DATASET_DIR, self.dataset_name.split("/")[-1]))

        local_path = None
        for p in possible_paths:
            if os.path.exists(p):
                local_path = p
                logger.info(f"Local dataset directory located at: {local_path}")
                break
        
        raw_datasets = None
        
        # Strategy 1: Load pre-saved Arrow format (preferred for speed).
        if local_path and (os.path.exists(os.path.join(local_path, "dataset_dict.json")) or \
                           os.path.exists(os.path.join(local_path, "state.json"))):
            logger.info("Detected 'save_to_disk' (Arrow) format. Executing load_from_disk.")
            try:
                raw_datasets = load_from_disk(local_path)
            except Exception as e:
                logger.warning(f"Failed to load from disk: {e}. Proceeding to next strategy.")

        # Strategy 2: Automatic Parquet File Detection.
        # This is required when datasets are downloaded purely as data files without a loading script.
        if raw_datasets is None and local_path:
            parquet_files = glob.glob(os.path.join(local_path, "**", "*.parquet"), recursive=True)
            
            if parquet_files:
                logger.info(f"Detected {len(parquet_files)} Parquet files. initializing Parquet builder.")
                data_files = {}
                
                # Heuristic mapping of filenames to standard splits.
                train_files = [f for f in parquet_files if "train" in os.path.basename(f).lower()]
                test_files = [f for f in parquet_files if "test" in os.path.basename(f).lower() or "val" in os.path.basename(f).lower()]
                
                if train_files:
                    data_files["train"] = train_files
                if test_files:
                    # Note: We tentatively map 'test'/'val' here; strict split validation occurs later.
                    data_files["test"] = test_files
                
                if data_files:
                    try:
                        raw_datasets = load_dataset("parquet", data_files=data_files)
                    except Exception as e:
                        logger.warning(f"Parquet loading failed: {e}")

        # Strategy 3: Standard Local Script Loading.
        if raw_datasets is None and local_path:
            logger.info(f"Attempting standard loading from local path: {local_path}")
            try:
                raw_datasets = load_dataset(local_path, self.config_name, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Standard local loading failed: {e}")

        # Strategy 4: Online Fallback (Hugging Face Hub).
        if raw_datasets is None:
            logger.info("Local strategies exhausted. Falling back to Hugging Face Hub (Online).")
            # Determine correct path logic for online retrieval
            path_to_load = os.path.join(DATASET_DIR, self.dataset_name) if os.path.exists(os.path.join(DATASET_DIR, self.dataset_name)) else self.dataset_name
            raw_datasets = load_dataset(path_to_load, self.config_name, trust_remote_code=True)
        
        # --- Split Validation and Assignment ---
        if "train" in raw_datasets:
            self.train_dataset = raw_datasets["train"]
        else:
            raise ValueError(f"Critical Error: Dataset missing 'train' split. Available keys: {raw_datasets.keys()}")
        
        # Standardize evaluation split naming (test vs. validation).
        if "test" in raw_datasets:
            self.eval_dataset = raw_datasets["test"]
        elif "validation" in raw_datasets:
            self.eval_dataset = raw_datasets["validation"]
        else:
            raise ValueError(f"Critical Error: Dataset must contain 'test' or 'validation' split. Available keys: {raw_datasets.keys()}")

        # --- Metadata Extraction (Label Mapping) ---
        potential_label_keys = ["label", "labels", "fine_label", "coarse_label"]
        label_key = next((k for k in potential_label_keys if k in self.train_dataset.features), None)
        
        if label_key:
            features = self.train_dataset.features[label_key]
            if hasattr(features, "names"):
                # Standard case: Features contain ClassLabel metadata.
                self.labels = features.names
                self.num_labels = len(self.labels)
                self.id2label = {i: label for i, label in enumerate(self.labels)}
                self.label2id = {label: i for i, label in enumerate(self.labels)}
                logger.info(f"Successfully extracted {self.num_labels} classes: {self.labels}")
                
                # Normalize column name to 'label' for downstream compatibility.
                if label_key != "label":
                    self.train_dataset = self.train_dataset.rename_column(label_key, "label")
                    self.eval_dataset = self.eval_dataset.rename_column(label_key, "label")
            else:
                # Edge Case: Metadata is missing (common with Parquet/Arrow raw loads).
                # Heuristic restoration for CIFAR-10 to prevent runtime failures.
                name_clean = self.dataset_name.lower().replace("-", "").replace("/", "")
                if "cifar10" in name_clean and self.num_labels == 0:
                    logger.info("Metadata missing. Applying manual restoration for CIFAR-10 labels.")
                    self.labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                    self.num_labels = 10
                    self.id2label = {i: label for i, label in enumerate(self.labels)}
                    self.label2id = {label: i for i, label in enumerate(self.labels)}
                else:
                    logger.warning("Label feature detected but lacks metadata (names). Manual mapping may be required.")
        else:
             logger.warning(f"Label column not found. Available features: {self.train_dataset.features.keys()}")

class CIFAR10Dataset(VisionDataset):
    """
    Wrapper for the CIFAR-10 dataset.
    Configured to match the directory structure: 'uoft-cs/cifar-10'.
    """
    def __init__(self):
        super().__init__(dataset_name="uoft-cs/cifar-10")

class CIFAR100Dataset(VisionDataset):
    """
    Wrapper for the CIFAR-100 dataset.
    Configured to match the directory structure: 'uoft-cs/cifar-100'.
    """
    def __init__(self):
        super().__init__(dataset_name="uoft-cs/cifar-100")

class ImageNetDataset(VisionDataset):
    """
    Wrapper for the ImageNet-1k dataset.
    """
    def __init__(self):
        super().__init__(dataset_name="imagenet-1k")

def get_vision_task(task_name: str) -> VisionDataset:
    """
    Factory function to instantiate the appropriate VisionDataset subclass.
    
    Args:
        task_name (str): The identifier for the dataset (e.g., 'cifar10', 'imagenet').
        
    Returns:
        VisionDataset: An initialized dataset object.
    """
    task_map = {
        "cifar10": CIFAR10Dataset,
        "uoft-cs/cifar10": CIFAR10Dataset,
        "uoft-cs/cifar-10": CIFAR10Dataset, # Support for hyphenated variations
        "cifar100": CIFAR100Dataset,
        "uoft-cs/cifar100": CIFAR100Dataset,
        "imagenet": ImageNetDataset,
        "imagenet-1k": ImageNetDataset,
    }
    
    normalized_name = task_name.lower().strip()
    
    if normalized_name in task_map:
        return task_map[normalized_name]()
    else:
        logger.info(f"Task '{task_name}' is not explicitly defined in the map. Attempting generic load.")
        return VisionDataset(dataset_name=task_name)