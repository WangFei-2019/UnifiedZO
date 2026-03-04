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
        核心数据构造契约：
        确保返回所有 UnifiedZO 评估所需的键值，包括用于 Variance Reduction 的 prompt 掩码。
        """
        # 1. 预处理图像并获取 pixel_values
        pixel_values = self.processor.image_processor(image, return_tensors="pt")["pixel_values"][0]
        
        # 2. 获取 Image Token ID (兼容 LLaVA-1.5 及后续变体)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if image_token_id is None or image_token_id == self.processor.tokenizer.unk_token_id:
             # Fallback: 32000 是 LLaVA 默认的视觉占位符 ID
             image_token_id = 32000 
        
        # 3. [架构优化] 动态计算 Image Token 数量
        # 取代硬编码的 576，根据视觉塔的实际输入尺寸和 patch_size 动态推导
        h, w = pixel_values.shape[1], pixel_values.shape[2]
        patch_size = getattr(self.processor.image_processor, 'patch_size', 14)
        num_image_tokens = (h // patch_size) * (w // patch_size)
        
        prefix_str = "USER: "
        suffix_str = f"\n{prompt_text}\nASSISTANT: "
        
        # 4. Tokenization (Prefix 带 BOS，Suffix/Answer 不带，实现无缝拼接)
        prefix_ids = self.processor.tokenizer(prefix_str, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        suffix_ids = self.processor.tokenizer(suffix_str, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        answer_ids = self.processor.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        
        # 构造图像占位符
        image_ids = torch.full((num_image_tokens,), image_token_id, dtype=torch.long)
        
        # 5. 拼接 Prompt 与 Full Input
        prompt_input_ids = torch.cat([prefix_ids, image_ids, suffix_ids])
        prompt_attention_mask = torch.ones_like(prompt_input_ids)
        
        full_input_ids = torch.cat([prompt_input_ids, answer_ids])
        full_attention_mask = torch.ones_like(full_input_ids)
        
        # 返回满足架构契约的字典
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "pixel_values": pixel_values,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask
        }

class ScienceQADataset(VLMBaseDataset):
    def __init__(self, 
                 split: str = "train", 
                 processor: transformers.ProcessorMixin = None,
                 data_dir: str = "/workspace/wangfei154/datasets/derek-thomas/ScienceQA"):
        super().__init__()
        logger.info(f"Loading ScienceQA from {data_dir} (split: {split})")
        # 离线环境强制加载本地数据
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
        防卫式数据整理：
        1. 检查 Schema 完整性，防止上游 remove_unused_columns 破坏数据。
        2. 动态处理 Padding 对齐。
        """
        if not instances:
            return {}

        # [架构补丁]：校验关键键值是否存在，Fail-Fast 拦截 KeyError
        required_keys = ["input_ids", "attention_mask", "pixel_values", "prompt_input_ids"]
        for k in required_keys:
            if k not in instances[0]:
                raise ValueError(
                    f"Architecture Error: Key '{k}' missing in Collator. "
                    f"Check if 'remove_unused_columns' is False in run_vlm.py."
                )

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
            
            # 执行 Padding 与 Label Masking (标签平移)
            if side == "left":
                padded_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), ids])
                padded_mask = torch.cat([torch.zeros((pad_len,), dtype=torch.long), mask])
                label = padded_ids.clone()
                # 只有 Answer Token 参与 Loss 计算
                label[:pad_len + prompt_len] = -100 
            else:
                padded_ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                padded_mask = torch.cat([mask, torch.zeros((pad_len,), dtype=torch.long)])
                label = padded_ids.clone()
                label[:prompt_len] = -100
                if pad_len > 0:
                    label[-pad_len:] = -100
                    
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