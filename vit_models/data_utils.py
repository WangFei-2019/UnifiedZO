import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize, 
    RandomResizedCrop, 
    RandomHorizontalFlip
)
from typing import Dict, Any, Union, List, Optional
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class VisionDatasetWrapper(Dataset):
    """
    A PyTorch Dataset wrapper that processes raw images into model inputs.

    This class handles the transformation pipeline:
    1. Retrieval of raw images from the Hugging Face dataset.
    2. Application of data augmentation (training) or deterministic preprocessing (evaluation).
    3. Conversion to 'pixel_values' tensors expected by ViT-like architectures.

    Attributes:
        dataset (Dataset): The underlying Hugging Face dataset containing raw images and labels.
        processor (AutoImageProcessor): The Hugging Face image processor (feature extractor) 
                                        associated with the pre-trained model.
        args (Namespace): Configuration arguments containing training hyperparameters.
        is_training (bool): Flag to determine whether to apply data augmentation.
    """
    def __init__(self, dataset, processor, args, is_training: bool = False):
        self.dataset = dataset
        self.processor = processor
        self.args = args
        self.is_training = is_training
        
        # Determine image key (e.g., 'img' for CIFAR, 'image' for ImageNet)
        self.image_key = "img" if "img" in dataset.features else "image"
        self.label_key = "label" if "label" in dataset.features else "labels"

        # Initialize Transforms
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        """
        Constructs the torchvision transform composition.

        Returns:
            Compose: A composition of image transformations.
        """
        # Retrieve image size from processor configuration
        # Robust handling for different Hugging Face processor versions
        size = None
        
        # Strategy 1: Check 'size' attribute (Standard in newer transformers)
        # Formats: {"height": 224, "width": 224} OR {"shortest_edge": 224}
        if hasattr(self.processor, "size"):
            if "height" in self.processor.size and "width" in self.processor.size:
                size = (self.processor.size["height"], self.processor.size["width"])
            elif "shortest_edge" in self.processor.size:
                # If only shortest_edge is defined, we typically assume a square crop for ViT
                s = self.processor.size["shortest_edge"]
                size = (s, s)

        # Strategy 2: Check 'crop_size' attribute (Legacy/Deprecated in some models)
        if size is None and hasattr(self.processor, "crop_size"):
            if "height" in self.processor.crop_size and "width" in self.processor.crop_size:
                size = (self.processor.crop_size["height"], self.processor.crop_size["width"])

        # Strategy 3: Fallback with Warning
        if size is None:
            logger.warning(f"Could not detect image size from processor config (checked 'size' and 'crop_size'). Defaulting to (224, 224).")
            size = (224, 224) 
        
        # Retrieve normalization statistics
        image_mean = getattr(self.processor, "image_mean", [0.5, 0.5, 0.5])
        image_std = getattr(self.processor, "image_std", [0.5, 0.5, 0.5])
        
        if self.is_training:
            # Data Augmentation for Training
            return Compose([
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=image_mean, std=image_std),
            ])
        else:
            # Deterministic Preprocessing for Evaluation
            return Compose([
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                Normalize(mean=image_mean, std=image_std),
            ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves and processes a single sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'pixel_values': Preprocessed image tensor (C, H, W).
                - 'labels': Class label index (LongTensor).
        """
        item = self.dataset[idx]
        
        # Extract image and ensure it's in RGB
        image = item[self.image_key]
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply transformations
        # Note: We use torchvision transforms manually here for flexibility, 
        # effectively replacing the processor's internal __call__ for single items.
        pixel_values = self.transforms(image)

        # Prepare label
        label = item[self.label_key]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }

def process_vision_dataset(
    args, 
    hf_dataset, 
    processor, 
    is_training: bool = True
) -> VisionDatasetWrapper:
    """
    Factory function to instantiate the VisionDatasetWrapper.

    Args:
        args (Namespace): Training arguments.
        hf_dataset (Dataset): The raw Hugging Face dataset split.
        processor (AutoImageProcessor): Model-specific image processor.
        is_training (bool): Whether the dataset is for training (enables augmentation).

    Returns:
        VisionDatasetWrapper: The wrapped PyTorch dataset ready for the DataLoader.
    """
    return VisionDatasetWrapper(hf_dataset, processor, args, is_training)