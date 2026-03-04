import torch
from dataclasses import dataclass
from typing import Dict, Sequence, List
import transformers
from torch.utils.data import Dataset

class ScienceQADataset(Dataset):
    """
    PyTorch Dataset for the ScienceQA Multimodal Benchmark.
    Specifically filtered for image-based questions to evaluate VLM capabilities.
    The prompt is formatted to restrict the output space to multiple-choice letters (A, B, C, D)
    to facilitate faster convergence during Zeroth-Order (ZO) Optimization.
    """
    def __init__(self, split: str = "train", processor: transformers.ProcessorMixin = None):
        super().__init__()
        print(f"Loading ScienceQA dataset (split: {split})...")
        # Load the official ScienceQA dataset
        raw_dataset = load_dataset("derek-thomas/ScienceQA", split=split)
        
        # Filter: We ONLY care about instances that contain an image for VLM evaluation
        self.dataset = raw_dataset.filter(lambda x: x['image'] is not None)
        print(f"Filtered image-based questions: {len(self.dataset)} samples.")
        
        self.processor = processor
        self.choice_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        
        # 1. Extract and process the PIL Image
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # 2. Extract textual components
        question = item['question']
        choices = item['choices']
        hint = item['hint'] if item['hint'] is not None else ""
        answer_idx = item['answer']
        
        # 3. Construct the Multiple-Choice Options String
        # E.g., "(A) gravity (B) friction (C) magnetism"
        options_str = " ".join([f"({self.choice_map[i]}) {c}" for i, c in enumerate(choices)])
        
        # 4. Construct the Multimodal Prompt
        # Note: The "<image>" token is a standardized placeholder for most VLMs (like LLaVA)
        # to inject visual patch embeddings.
        prompt = f"<image>\nContext: {hint}\nQuestion: {question}\nOptions: {options_str}\nAnswer:"
        
        # 5. Construct the Target Output
        # Restricting the target to just the letter greatly reduces the learning difficulty for ZO
        target = f" ({self.choice_map[answer_idx]})"
        
        # For standard Causal LM training, the text is the concatenation of prompt and target.
        # The DataCollator will tokenize this complete string.
        full_text = prompt + target
        
        return {
            "text": full_text,
            "images": image
        }
    
@dataclass
class DataCollatorForVLM:
    """
    Collate function for vision-language models.
    It pads text inputs to the maximum length of the batch and stacks image pixel values.
    """
    processor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. Extract texts and apply text padding
        texts = [instance['text'] for instance in instances]
        text_inputs = self.processor.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            truncation=True
        )
        
        batch = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": text_inputs["input_ids"].clone() # Standard Causal LM objective
        }
        
        # 2. Extract and stack image tensors if present
        if 'images' in instances[0] and instances[0]['images'] is not None:
            images = [instance['images'] for instance in instances]
            # processor.image_processor handles resizing, normalization, and conversion to tensor
            image_inputs = self.processor.image_processor(images, return_tensors="pt")
            batch["pixel_values"] = image_inputs["pixel_values"]
            
            # Optionally handle image sizes for models like LLaVA-NeXT or Qwen-VL
            if "image_sizes" in image_inputs:
                 batch["image_sizes"] = image_inputs["image_sizes"]
                 
        return batch