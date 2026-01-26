import os
import os.path as osp
import argparse
import logging
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from gptqmodel import GPTQModel, QuantizeConfig

# Initialize logging configuration
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define dataset directory from environment variable or default path
DATASET_DIR = os.getenv("DATASET_DIR", "/workspace/wangfei154/datasets")

def get_calib_dataset(processor, dataset_name="uoft-cs/cifar-10", n_samples=128):
    """
    Loads and preprocesses a calibration dataset for Post-Training Quantization (PTQ).

    According to standard literature (e.g., FQ-ViT, PTQ4ViT), a subset of the 
    training set (usually 128-1024 samples) is used to estimate quantization parameters.

    Args:
        processor: The image processor associated with the ViT model.
        dataset_name (str): Path or name of the dataset (e.g., 'imagenet-1k' or 'uoft-cs/cifar-10').
        n_samples (int): Number of calibration samples to load (default: 128).

    Returns:
        List[torch.Tensor]: A list of preprocessed pixel_values tensors.
    """
    logger.info(f"Loading calibration dataset: {dataset_name} (Split: train, Samples: {n_samples})...")
    
    # Attempt to resolve local path for the dataset to avoid re-downloading
    local_path = dataset_name
    if DATASET_DIR:
        potential_path = os.path.join(DATASET_DIR, dataset_name)
        if os.path.exists(potential_path):
            local_path = potential_path
            logger.info(f"Found local dataset at: {local_path}")

    try:
        # Load the training split and shuffle to get a random subset
        dataset = load_dataset(local_path, split="train")
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
    except Exception as e:
        logger.error(f"Failed to load dataset {local_path}: {e}")
        raise e
    
    samples = []
    for item in dataset:
        # Handle varying column names (e.g., 'img' for CIFAR, 'image' for ImageNet)
        image = item.get("img", item.get("image"))
        
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Preprocess image: output is a dict {'pixel_values': tensor}
        # return_tensors="pt" ensures PyTorch tensor output
        inputs = processor(images=image, return_tensors="pt")
        samples.append(inputs.pixel_values) 

    return samples

def run_gptq(args):
    """
    Executes the GPTQ quantization process for Vision Transformers.
    """
    model_path = args.model_path
    quant_path = args.quant_path
    bits = args.bits
    
    logger.info(f"Loading Image Processor from {model_path}...")
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # 1. Prepare Calibration Data
    # Ideally, use 'imagenet-1k' for base models. Fallback to 'cifar-10' if unavailable.
    # We use the user's specific dataset path by default.
    # calib_dataset_name = "uoft-cs/cifar-10" 
    # Uncomment the following line if ImageNet is available:
    calib_dataset_name = "imagenet-1k"
    
    pixel_values_list = get_calib_dataset(processor, dataset_name=calib_dataset_name, n_samples=128)
    
    # 2. Initialize Quantization Configuration
    logger.info(f"Initializing GPTQ configuration with {bits}-bit precision...")
    quant_config = QuantizeConfig(
        bits=bits, 
        group_size=128,
        # 'desc_act' (activation reordering) can sometimes be unstable for ViTs; 
        # set to False for robust base quantization.
        desc_act=False  
    )
    
    # 3. Load Model with Quantization Support
    logger.info(f"Loading model: {model_path}")
    # Note: We rely on 'trust_remote_code=True' for custom model architectures.
    model = GPTQModel.from_pretrained(
        model_path,
        quant_config,
        trust_remote_code=True,
        device_map="auto"
    )

    # 4. Execute Quantization
    logger.info("Starting quantization process (this may take a while)...")
    # ViT models expect 'pixel_values', so we pass the prepared list of tensors.
    model.quantize(pixel_values_list, batch_size=1)
    
    # 5. Save Quantized Model
    model_name = model_path.rstrip('/').split('/')[-1]
    suffix = f'-gptq-b{bits}-g128'
    
    if not os.path.exists(quant_path):
        os.makedirs(quant_path)
        
    final_save_path = osp.join(quant_path, model_name + suffix)
    
    logger.info(f"Saving quantized model and processor to {final_save_path}...")
    model.save(final_save_path)
    processor.save_pretrained(final_save_path)
    
    logger.info("Quantization completed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Post-Training Quantization for ViT using GPTQ')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the full-precision model (e.g., google/vit-base-patch16-224)')
    parser.add_argument('--quant_path', type=str, default='quantized_models', help='Directory to save the quantized model')
    parser.add_argument('--bits', type=int, default=4, help='Target bit-width (e.g., 4 or 8)')
    
    args = parser.parse_args()
    run_gptq(args)

if __name__ == '__main__':
    main()