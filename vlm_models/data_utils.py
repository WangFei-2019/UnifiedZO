import torch
from dataclasses import dataclass
from typing import Dict, Sequence, List
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class VLMBaseDataset(Dataset):
    def _apply_chat_template_and_mask(self, image: Image.Image, prompt_text: str, answer_text: str):
        """
        Prepares multimodal inputs conforming strictly to Hugging Face's LLaVA architecture.
        Delegates <image> token expansion to the official processor to prevent token-feature mismatch.
        """
        # 1. Standardize prompt
        prompt_str = f"USER: <image>\n{prompt_text}\nASSISTANT: "
        
        # 2. Let the processor automatically expand <image> to 576 tokens
        # and generate the corresponding pixel_values simultaneously.
        prompt_inputs = self.processor(text=prompt_str, images=image, return_tensors="pt")
        
        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_mask = prompt_inputs["attention_mask"][0]
        pixel_values = prompt_inputs["pixel_values"][0]
        
        # 3. Tokenize the answer text seamlessly (without redundant BOS tokens)
        answer_ids = self.processor.tokenizer(
            answer_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        answer_mask = torch.ones_like(answer_ids)
        
        # 4. Concatenate prompt and answer
        full_input_ids = torch.cat([prompt_ids, answer_ids])
        full_attention_mask = torch.cat([prompt_mask, answer_mask])
        
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "pixel_values": pixel_values,
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask
        }

class ScienceQADataset(VLMBaseDataset):
    def __init__(self, 
                 split: str = "train", 
                 processor: transformers.ProcessorMixin = None,
                 data_dir: str = "/workspace/wangfei154/datasets/derek-thomas/ScienceQA"):
        super().__init__()
        logger.info(f"Loading ScienceQA from {data_dir} (split: {split})")

        raw_dataset = load_dataset(data_dir, split=split)
        
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
    def __init__(self, 
                 split: str = "train", 
                 processor: transformers.ProcessorMixin = None,
                 data_dir: str = "/workspace/wangfei154/datasets/AI4Math/MathVista"):
        super().__init__()
        logger.info(f"Loading MathVista from {data_dir} (split: {split})")
        raw_dataset = load_dataset(data_dir, split=split)
        
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

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Dynamic padding and label masking collator.
        Properly ignores prompt tokens and dynamically pads answers.
        """
        if not instances:
            return {}

        required_keys = ["input_ids", "attention_mask", "pixel_values", "prompt_input_ids"]
        for k in required_keys:
            if k not in instances[0]:
                raise ValueError(f"Architecture Error: Key '{k}' missing. Set remove_unused_columns=False.")

        input_ids = [inst["input_ids"] for inst in instances]
        attention_mask = [inst["attention_mask"] for inst in instances]
        pixel_values = [inst["pixel_values"] for inst in instances]
        prompt_input_ids = [inst["prompt_input_ids"] for inst in instances]
        
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
            
        side = getattr(self.processor.tokenizer, 'padding_side', 'right') # Defaulting to right for generation safety
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
            
            # Pad and construct labels
            if side == "left":
                padded_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), ids])
                padded_mask = torch.cat([torch.zeros((pad_len,), dtype=torch.long), mask])
                label = padded_ids.clone()
                label[:pad_len + prompt_len] = -100 # Mask padding and prompt
            else:
                padded_ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                padded_mask = torch.cat([mask, torch.zeros((pad_len,), dtype=torch.long)])
                label = padded_ids.clone()
                label[:prompt_len] = -100 # Mask prompt
                if pad_len > 0:
                    label[-pad_len:] = -100 # Mask padding
                    
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
            
        return batch