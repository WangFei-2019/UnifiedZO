import logging
from datasets import load_dataset
from typing import Dict, Any, Optional

# Initialize logger for the module
logger = logging.getLogger(__name__)

class VisionDataset:
    """
    Base class for vision datasets handling data loading and metadata extraction.

    This class serves as a wrapper around Hugging Face datasets, standardizing the 
    interface for retrieving training/validation splits and label taxonomies 
    (id2label and label2id mappings) required for model initialization.

    Attributes:
        dataset_name (str): The identifier of the dataset on Hugging Face Hub.
        config_name (Optional[str]): The specific configuration of the dataset (e.g., subset name).
        num_labels (int): The total number of classification categories.
        id2label (Dict[int, str]): Mapping from integer class indices to string class names.
        label2id (Dict[str, int]): Mapping from string class names to integer class indices.
        train_dataset (Dataset): The training data split.
        eval_dataset (Dataset): The evaluation (validation or test) data split.
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
        Loads the dataset from the Hugging Face Hub and initializes label mappings.

        This method performs two primary functions:
        1. Downloads and caches the dataset splits (Train/Test/Validation).
        2. Inspects the dataset features to automatically extract the label schema
           (ClassLabel), ensuring compatibility with the model's classification head.
        
        Raises:
            ValueError: If the dataset does not contain a valid evaluation split 
            ('test' or 'validation').
        """
        logger.info(f"Loading vision dataset: {self.dataset_name}")
        
        # Load the raw dataset from Hugging Face
        # trust_remote_code=True is often required for custom datasets
        raw_datasets = load_dataset(self.dataset_name, self.config_name, trust_remote_code=True)
        
        # Assign Training Split
        if "train" in raw_datasets:
            self.train_dataset = raw_datasets["train"]
        else:
            raise ValueError(f"Dataset {self.dataset_name} is missing a 'train' split.")
        
        # Assign Evaluation Split (Prioritize 'test', fall back to 'validation')
        if "test" in raw_datasets:
            self.eval_dataset = raw_datasets["test"]
        elif "validation" in raw_datasets:
            self.eval_dataset = raw_datasets["validation"]
        else:
            raise ValueError(f"Dataset {self.dataset_name} must have 'test' or 'validation' split.")

        # Extract Label Metadata
        # We assume the dataset has a 'label' or 'labels' feature which is of type ClassLabel.
        # This is standard for datasets like CIFAR, ImageNet, FashionMNIST.
        label_key = "label" if "label" in self.train_dataset.features else "labels"
        
        if label_key in self.train_dataset.features:
            features = self.train_dataset.features[label_key]
            if hasattr(features, "names"):
                self.labels = features.names
                self.num_labels = len(self.labels)
                self.id2label = {i: label for i, label in enumerate(self.labels)}
                self.label2id = {label: i for i, label in enumerate(self.labels)}
                logger.info(f"Detected {self.num_labels} classes: {self.labels}")
            else:
                logger.warning("Label feature does not have 'names' attribute. Manual label mapping may be required.")
        else:
            logger.warning(f"Could not find '{label_key}' in dataset features. Ensure the dataset is a classification task.")

class CIFAR10Dataset(VisionDataset):
    """
    Handler for the CIFAR-10 dataset.
    
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
    with 6000 images per class.
    
    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    def __init__(self):
        super().__init__(dataset_name="cifar10")

class CIFAR100Dataset(VisionDataset):
    """
    Handler for the CIFAR-100 dataset.
    
    This dataset is just like the CIFAR-10, except it has 100 classes containing 
    600 images each.
    """
    def __init__(self):
        super().__init__(dataset_name="cifar100")

class ImageNetDataset(VisionDataset):
    """
    Handler for the ImageNet-1k dataset.
    
    Standard benchmark dataset for image classification. Note: This requires 
    access to the dataset files or a logged-in Hugging Face account with access.
    """
    def __init__(self):
        super().__init__(dataset_name="imagenet-1k")

def get_vision_task(task_name: str) -> VisionDataset:
    """
    Factory method to instantiate the appropriate VisionDataset object.

    Args:
        task_name (str): The name of the task/dataset (e.g., 'cifar10', 'imagenet').

    Returns:
        VisionDataset: An initialized instance of the requested dataset wrapper.
    """
    task_map = {
        "cifar10": CIFAR10Dataset,
        "cifar100": CIFAR100Dataset,
        "imagenet": ImageNetDataset,
        # Add more task mappings here as needed
    }
    
    normalized_name = task_name.lower().strip()
    
    if normalized_name in task_map:
        return task_map[normalized_name]()
    else:
        # Fallback: Try to load inputs as a generic Hugging Face dataset name
        logger.info(f"Task '{task_name}' not explicitly defined. Attempting to load as generic HF dataset.")
        return VisionDataset(dataset_name=task_name)