import torch
from dataclasses import dataclass
from typing import Dict, Sequence
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class VLMBaseDataset(Dataset):
    def _apply_chat_template_and_mask(self, image: Image.Image, prompt_text: str, answer_text: str):
        
        pixel_values = self.processor.image_processor(image, return_tensors="pt")["pixel_values"][0]
        
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if image_token_id is None or image_token_id == self.processor.tokenizer.unk_token_id:
             # Fallback for some specific LLaVA tokenizers
             image_token_id = 32000 
        
        prefix_str = "USER: "
        suffix_str = f"\n{prompt_text}\nASSISTANT: "
        
        prefix_ids = self.processor.tokenizer(prefix_str, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        suffix_ids = self.processor.tokenizer(suffix_str, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        answer_ids = self.processor.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        
        num_image_tokens = 576
        image_ids = torch.full((num_image_tokens,), image_token_id, dtype=torch.long)
        
        prompt_input_ids = torch.cat([prefix_ids, image_ids, suffix_ids])
        prompt_attention_mask = torch.ones_like(prompt_input_ids)
        
        full_input_ids = torch.cat([prompt_input_ids, answer_ids])
        full_attention_mask = torch.ones_like(full_input_ids)
        
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "pixel_values": pixel_values,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask
        }

class ScienceQADataset(VLMBaseDataset):
    def __init__(self, split: str = "train", processor: transformers.ProcessorMixin = None):
        super().__init__()
        logger.info(f"Loading ScienceQA dataset (split: {split})...")
        raw_dataset = load_dataset("/workspace/wangfei154/datasets/derek-thomas/ScienceQA", split=split)
        self.dataset = raw_dataset.filter(lambda x: x['image'] is not None)
        self.processor = processor
        self.choice_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        image = item['image'].convert('RGB') if item['image'].mode != 'RGB' else item['image']
        choices_str = " ".join([f"({self.choice_map[i]}) {c}" for i, c in enumerate(item['choices'])])
        hint = item['hint'] if item['hint'] is not None else ""
        prompt = f"Context: {hint}\nQuestion: {item['question']}\nOptions: {choices_str}\nAnswer with only the option letter."
        answer = f"({self.choice_map[item['answer']]})"
        return self._apply_chat_template_and_mask(image, prompt, answer)

class MathVistaDataset(VLMBaseDataset):
    def __init__(self, split: str = "train", processor: transformers.ProcessorMixin = None):
        super().__init__()
        logger.info(f"Loading MathVista dataset (split: {split}) for complex reasoning evaluation...")
        raw_dataset = load_dataset("/workspace/wangfei154/datasets/AI4Math/MathVista", split=split)
        self.dataset = raw_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        image = item['decoded_image'].convert('RGB')
        prompt = f"Analyze the image and solve the mathematical question.\nQuestion: {item['query']}\nAnswer:"
        answer = str(item['answer'])
        return self._apply_chat_template_and_mask(image, prompt, answer)

@dataclass
class DataCollatorForVLM:
    processor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        attention_mask = [inst["attention_mask"] for inst in instances]
        pixel_values = [inst["pixel_values"] for inst in instances]
        prompt_input_ids = [inst["prompt_input_ids"] for inst in instances]
        
        pad_token_id = self.processor.tokenizer.pad_token_id
        side = self.processor.tokenizer.padding_side
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids, padded_attention_mask, labels, option_lens = [], [], [], []

        for i in range(len(instances)):
            ids = input_ids[i]
            mask = attention_mask[i]
            p_ids = prompt_input_ids[i]
            
            seq_len = len(ids)
            prompt_len = len(p_ids)
            ans_len = seq_len - prompt_len
            option_lens.append(ans_len)
            
            pad_len = max_len - seq_len
            
            if side == "left":
                padded_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), ids])
                padded_mask = torch.cat([torch.zeros((pad_len,), dtype=torch.long), mask])
                label = padded_ids.clone()
                label[:pad_len + prompt_len] = -100 # Precise Masking: Padding + Prompt
            else:
                padded_ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                padded_mask = torch.cat([mask, torch.zeros((pad_len,), dtype=torch.long)])
                label = padded_ids.clone()
                label[:prompt_len] = -100 # Precise Masking: Prompt
                if pad_len > 0:
                    label[-pad_len:] = -100 # Precise Masking: Right Padding
                    
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            labels.append(label)

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(labels),
            "pixel_values": torch.stack(pixel_values),
            "option_len": torch.tensor(option_lens, dtype=torch.long)
        }
        if "image_sizes" in instances[0]:
            batch["image_sizes"] = torch.stack([inst["image_sizes"] for inst in instances])
            
        return batch