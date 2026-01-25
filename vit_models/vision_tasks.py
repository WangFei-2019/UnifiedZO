import logging
import os
import glob
from datasets import load_dataset, load_from_disk
from typing import Dict, Any, Optional

# Initialize logger for the module
logger = logging.getLogger(__name__)

# --- Configure your dataset root directory ---
# Ensure that the 'uoft-cs/cifar10' or 'cifar10' folder exists under this path
DATASET_DIR = "/workspace/wangfei154/datasets/" 

class VisionDataset:
    """
    Base class for vision datasets handling data loading and metadata extraction.
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
        Loads the dataset with offline support.
        Priority:
        1. Local path via load_from_disk (Arrow format)
        2. Local path via load_dataset("parquet") (Parquet format - Target case)
        3. Local path via load_dataset (Script format with .py file)
        4. Hugging Face Hub (Online)
        """
        logger.info(f"Loading vision dataset: {self.dataset_name}")
        
        # 1. Attempt to construct local path
        # Compatible with two structures: datasets/cifar10 or datasets/uoft-cs/cifar10
        possible_paths = []
        if DATASET_DIR:
            possible_paths.append(os.path.join(DATASET_DIR, self.dataset_name))
            # If the name contains a slash (e.g., uoft-cs/cifar10), also try using just the suffix
            if "/" in self.dataset_name:
                possible_paths.append(os.path.join(DATASET_DIR, self.dataset_name.split("/")[-1]))

        local_path = None
        for p in possible_paths:
            if os.path.exists(p):
                local_path = p
                logger.info(f"Found local dataset folder at: {local_path}")
                break
        
        raw_datasets = None
        
        # Strategy 1: load_from_disk (Arrow/save_to_disk format)
        if local_path and (os.path.exists(os.path.join(local_path, "dataset_dict.json")) or \
                           os.path.exists(os.path.join(local_path, "state.json"))):
            logger.info("Detected 'save_to_disk' format. Using load_from_disk.")
            try:
                raw_datasets = load_from_disk(local_path)
            except Exception as e:
                logger.warning(f"load_from_disk failed: {e}")

        # Strategy 1.5: Parquet Files (Automatic Detection)
        # Automatically scan for .parquet files in the directory
        if raw_datasets is None and local_path:
            # Search for parquet files recursively or in the current directory
            parquet_files = glob.glob(os.path.join(local_path, "**", "*.parquet"), recursive=True)
            
            if parquet_files:
                logger.info(f"Detected {len(parquet_files)} parquet files. Using 'parquet' builder.")
                data_files = {}
                
                # Simple keyword matching logic for splits
                train_files = [f for f in parquet_files if "train" in os.path.basename(f).lower()]
                test_files = [f for f in parquet_files if "test" in os.path.basename(f).lower() or "val" in os.path.basename(f).lower()]
                
                if train_files:
                    data_files["train"] = train_files
                if test_files:
                    # CIFAR10 usually names it 'test', but HF datasets standard often maps it to 'validation'.
                    # We load them here and handle the split naming later.
                    data_files["test"] = test_files
                
                if data_files:
                    try:
                        # Core modification: Specify engine="pyarrow" and input as "parquet" to load raw files
                        raw_datasets = load_dataset("parquet", data_files=data_files)
                    except Exception as e:
                        logger.warning(f"Failed to load parquet files: {e}")

        # Strategy 2: Local Script (If a .py script exists)
        if raw_datasets is None and local_path:
            logger.info(f"Attempting standard load_dataset from: {local_path}")
            try:
                raw_datasets = load_dataset(local_path, self.config_name, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Standard local load failed (expected if no .py script): {e}")

        # Strategy 3: Fallback to Online (Hugging Face Hub)
        if raw_datasets is None:
            logger.info("Falling back to Hugging Face Hub (Online)...")
            raw_datasets = load_dataset(os.path.join(DATASET_DIR, self.dataset_name), self.config_name, trust_remote_code=True)
        
        # --- Split Assignment ---
        if "train" in raw_datasets:
            self.train_dataset = raw_datasets["train"]
        else:
            raise ValueError(f"Dataset {self.dataset_name} loaded but missing 'train' split. Available: {raw_datasets.keys()}")
        
        # Assign Evaluation Split
        if "test" in raw_datasets:
            self.eval_dataset = raw_datasets["test"]
        elif "validation" in raw_datasets:
            self.eval_dataset = raw_datasets["validation"]
        else:
            # If neither test nor validation exists (e.g., parquet matching failed), raise error
            raise ValueError(f"Dataset must have 'test' or 'validation' split. Available: {raw_datasets.keys()}")

        # --- Label Metadata Extraction ---
        potential_label_keys = ["label", "labels", "fine_label", "coarse_label"]
        label_key = next((k for k in potential_label_keys if k in self.train_dataset.features), None)
        
        if label_key:
            features = self.train_dataset.features[label_key]
            if hasattr(features, "names"):
                self.labels = features.names
                self.num_labels = len(self.labels)
                self.id2label = {i: label for i, label in enumerate(self.labels)}
                self.label2id = {label: i for i, label in enumerate(self.labels)}
                logger.info(f"Detected {self.num_labels} classes: {self.labels}")
                
                if label_key != "label":
                    self.train_dataset = self.train_dataset.rename_column(label_key, "label")
                    self.eval_dataset = self.eval_dataset.rename_column(label_key, "label")
            else:
                # When loading via Parquet, Label is often just Int, losing names metadata.
                # For CIFAR-10, we can manually hardcode to fix and prevent errors.
                if "cifar10" in self.dataset_name.lower() and self.num_labels == 0:
                    logger.info("Restoring missing CIFAR-10 label metadata...")
                    self.labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                    self.num_labels = 10
                    self.id2label = {i: label for i, label in enumerate(self.labels)}
                    self.label2id = {label: i for i, label in enumerate(self.labels)}
                else:
                    logger.warning("Label feature exists but lacks metadata. Manual mapping might be needed.")
        else:
             # If the label column is completely missing in Parquet (rare), try hardcoding based on dataset name
             logger.warning(f"Could not find label column. Features: {self.train_dataset.features.keys()}")

class CIFAR10Dataset(VisionDataset):
    def __init__(self):
        # Use short name to match local directory structure
        super().__init__(dataset_name="cifar10")

class CIFAR100Dataset(VisionDataset):
    def __init__(self):
        super().__init__(dataset_name="cifar100")

class ImageNetDataset(VisionDataset):
    def __init__(self):
        super().__init__(dataset_name="imagenet-1k")

def get_vision_task(task_name: str) -> VisionDataset:
    task_map = {
        "cifar10": CIFAR10Dataset,
        "uoft-cs/cifar10": CIFAR10Dataset, 
        "cifar100": CIFAR100Dataset,
        "uoft-cs/cifar100": CIFAR100Dataset,
        "imagenet": ImageNetDataset,
        "imagenet-1k": ImageNetDataset,
    }
    
    normalized_name = task_name.lower().strip()
    
    if normalized_name in task_map:
        return task_map[normalized_name]()
    else:
        logger.info(f"Task '{task_name}' not explicitly defined. Attempting generic load.")
        return VisionDataset(dataset_name=task_name)