import os
import os.path as osp
import argparse
import logging
from typing import List, Union

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define dataset directory if needed, otherwise rely on HF cache
DATASET_DIR = os.getenv("DATASET_DIR", "/workspace/wangfei154/datasets")

def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=128,
    max_seq_len=512,
    split="train",
    text_column="text",
):
    """
    Prepare calibration dataset for quantization.
    Adapted from AWQ repository and LQZO implementation.
    """
    if isinstance(data, str):
        if data == "pileval":
            # Use local path if DATASET_DIR is set, else download from HF Hub
            path = "mit-han-lab/pile-val-backup"
            if DATASET_DIR:
                path = os.path.join(DATASET_DIR, path)
            dataset = load_dataset(path, split="validation")
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for item in dataset:
        if isinstance(item, list):
            line_encoded = item
        else:
            line = item[text_column]
            # Ensure line is a string
            if not isinstance(line, str):
                continue
            line = line.strip()
            line_encoded = tokenizer.encode(line)
            
        if len(line_encoded) > max_seq_len:
            continue
            
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
            
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
            
    # Concatenate all samples and split according to max sequence length
    if not samples:
        raise ValueError("No valid samples found in the dataset.")
        
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logger.debug(f" * Split into {n_split} blocks")
    
    return [
        cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
    ]


def run_gptq(args):
    """
    Execute GPTQ quantization process.
    """
    model_path = args.model_path
    quant_path = args.quant_path
    bits = args.bits
    
    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    logger.info("Preparing calibration dataset (Pile-val)...")
    # For GPTQ, we utilize PilEval dataset for calibration
    calibration_dataset = get_calib_dataset(data="pileval", tokenizer=tokenizer)

    # Preprocess dataset into a list of lists for GPTQModel
    processed_calibration_dataset = []
    for idx in range(len(calibration_dataset)):
        processed_calibration_dataset.append(calibration_dataset[idx][0].tolist())

    logger.info(f"Initializing GPTQModel with {bits}-bit quantization...")
    quant_config = QuantizeConfig(bits=bits, group_size=128) # Default group size 128
    
    model = GPTQModel.from_pretrained(
        model_path, 
        quant_config, 
        trust_remote_code=True, 
        device_map='auto'
    )

    logger.info("Starting quantization...")
    
    # Try quantize with different argument sets to handle version differences
    try:
        # removed calibration_enable_gpu_cache=True as it is not supported in newer versions
        model.quantize(
            processed_calibration_dataset, 
            tokenizer=tokenizer, 
            batch_size=2
        )
    except TypeError as e:
        # Fallback: some versions might not accept 'tokenizer' as argument if dataset is pre-processed
        if "tokenizer" in str(e):
            logger.warning(f"Quantization with tokenizer arg failed: {e}. Retrying without tokenizer...")
            model.quantize(
                processed_calibration_dataset, 
                batch_size=2
            )
        else:
            raise e
    
    # Construct save path
    model_name = model_path.rstrip('/').split('/')[-1]
    suffix = '-gptq' + f'-b{bits}' + '-g128'
    
    if not os.path.exists(quant_path):
        os.makedirs(quant_path)
        
    final_save_path = osp.join(quant_path, model_name + suffix)
    
    logger.info(f"Saving quantized model to {final_save_path}...")
    model.save(final_save_path)
    logger.info("Quantization completed successfully.")
    return

def main():
    parser = argparse.ArgumentParser(description='Quantization with Transformers')
    parser.add_argument('--model_path', type=str, default='facebook/opt-125m', help='Path to the model to quantize')
    parser.add_argument('--quant_mode', type=str, choices=['gptq'], default='gptq', help='Quantization mode')
    parser.add_argument('--quant_path', type=str, default='quantized_models', help='Directory to save the quantized model')
    parser.add_argument('--bits', type=int, default=4, help='Number of bits (e.g., 2, 4 or 8)')
    # parser.add_argument('--zero_point', type=bool, action='store_true', default=True)
    
    args = parser.parse_args()

    if args.quant_mode == 'gptq':
        run_gptq(args)
    else:
        raise NotImplementedError(f"Quantization mode {args.quant_mode} is not implemented.")
    
if __name__ == '__main__':
    main()