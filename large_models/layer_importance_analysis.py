# import os
# import sys
# import torch
# import json
# import re
# import numpy as np
# from tqdm import tqdm
# from collections import defaultdict
# from transformers import (
#     HfArgumentParser, 
#     AutoConfig, 
#     AutoTokenizer, 
#     AutoModelForCausalLM, 
#     set_seed,
#     DataCollatorForTokenClassification
# )

# # 引入本地模块
# sys.path.append(os.getcwd())
# from arguments import LMZOTrainingArguments
# from tasks import get_task
# from utils import process_dataset
# from zo_core.trainer.mezo_trainer import MeZOTrainer

# # =============================================================================
# # Helper Class: 继承 MeZOTrainer 以复用其核心扰动逻辑
# # =============================================================================
# class AnalysisMeZOTrainer(MeZOTrainer):
#     """
#     专门用于分析的 MeZO Trainer。
#     它复用了 MeZOTrainer 的 _perturb_mezo 和 zo_forward 方法，
#     但不执行参数更新，而是返回这一步的 projected_grad 和 seed。
#     """
#     def get_zo_grad_info(self, model, inputs):
#         """
#         执行一次标准的 MeZO 估算步骤 (Forward difference)，
#         但不更新参数。
#         """
#         model.eval()

#         # 1. Sample Seed
#         # MeZO 使用 np.random 生成种子
#         self.zo_random_seed = np.random.randint(1000000000)
        
#         # 2. Forward + (Perturb weight by +z)
#         self._perturb_mezo(scaling_factor=1)
#         loss1 = self.zo_forward(model, inputs)
        
#         # 3. Forward - (From +1 to -1 requires -2 step)
#         self._perturb_mezo(scaling_factor=-2)
#         loss2 = self.zo_forward(model, inputs)
        
#         # 4. Calculate Projected Gradient Estimate
#         # projected_grad 是一个标量，代表梯度在 z 方向上的投影大小
#         projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
#         # 5. Restore parameters (From -1 to 0 requires +1)
#         self._perturb_mezo(scaling_factor=1)
        
#         # 注意：这里我们不调用 _update_mezo，因为我们要对比的是"同一状态"下的梯度特性
        
#         return projected_grad, self.zo_random_seed

# # =============================================================================
# # Main Analysis Logic
# # =============================================================================

# def get_layer_id(name):
#     match = re.search(r"\.(layers|block|h|blocks)\.(\d+)\.", name)
#     if match: return int(match.group(2))
#     if "embed" in name: return -1 
#     return 9999 

# def main():
#     # --- 1. Setup & Args ---
#     parser = HfArgumentParser((LMZOTrainingArguments,))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
#     else:
#         args = parser.parse_args_into_dataclasses()[0]

#     set_seed(args.seed)
    
#     # 强制设置 ZO 参数以确保 MeZO 逻辑正确
#     args.trainer = "mezo"
#     # 如果没有指定 eps，给一个默认值
#     if not hasattr(args, 'zo_eps') or args.zo_eps is None:
#         args.zo_eps = 1e-3
    
#     # --- 2. Model & Data ---
#     print(f"Loading model: {args.model_name}")
#     config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         config.pad_token_id = tokenizer.eos_token_id
        
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name, config=config, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True
#     )
    
#     # Data Setup
#     task = get_task(args.task_name)
#     # 使用较少的样本进行深入分析
#     train_samples = task.sample_train_sets(num_train=64, num_dev=1, num_eval=1, num_train_sets=1, seed=42)[0]
#     train_dataset = process_dataset(args, task, train_samples, tokenizer, is_training=True)
#     collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
#     dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, collate_fn=collator, shuffle=True)

#     # --- 3. Initialize Analysis Trainer ---
#     # 我们实例化自定义的 Trainer 来获得 MeZO 的能力
#     # 需要传入一个 dummy evaluator
#     analysis_trainer = AnalysisMeZOTrainer(
#         model=model,
#         args=args,
#         train_dataset=train_dataset, 
#         eval_dataset=None,
#         tokenizer=tokenizer,
#         data_collator=collator
#     )
    
#     # 初始化 MeZO 状态 (构建 named_parameters_to_optim 列表)
#     analysis_trainer.named_parameters_to_optim = []
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             analysis_trainer.named_parameters_to_optim.append((name, param))

#     # --- 4. 探针设置 (Probe Setup) ---
#     PROBE_SIZE = 2048
    
#     # 存储结果
#     # 1. 累积能量 (Energy): 累积梯度的模长 (Scalar Sum) -> 反映总变动幅度
#     fo_energy = defaultdict(float)
#     zo_energy = defaultdict(float)
    
#     # 2. 净位移 (Net Displacement): 累积梯度向量本身 (Vector Sum) -> 反映由于方向一致性产生的有效位移
#     # 使用 float32 避免溢出
#     fo_disp_vec = {} 
#     zo_disp_vec = {}
    
#     # 初始化向量存储
#     for name, param in model.named_parameters():
#         if not param.requires_grad: continue
#         layer_id = get_layer_id(name)
#         if layer_id not in fo_disp_vec:
#             fo_disp_vec[layer_id] = torch.zeros(PROBE_SIZE, device=param.device, dtype=torch.float32)
#             zo_disp_vec[layer_id] = torch.zeros(PROBE_SIZE, device=param.device, dtype=torch.float32)

#     num_steps = 1000
#     print(f"Running Real MeZO vs Adam Analysis for {num_steps} steps...")
    
#     model.train() # Enable Dropout etc.
#     iterator = iter(dataloader)
    
#     for step in tqdm(range(num_steps), desc="Analyzing Trajectory"):
#         try:
#             inputs = next(iterator)
#         except StopIteration:
#             iterator = iter(dataloader)
#             inputs = next(iterator)
        
#         # 移至设备
#         inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
#         # =================================================
#         # Part A: First-Order (Adam) Ground Truth
#         # =================================================
#         # 1. 标准反向传播
#         model.zero_grad()
#         outputs = model(**inputs)
#         loss = outputs.loss
#         loss.backward()
        
#         # 2. 记录 FO 数据
#         for name, param in model.named_parameters():
#             if param.grad is None: continue
#             layer_id = get_layer_id(name)
            
#             # 取出 Probe 部分
#             grad_flat = param.grad.data.view(-1)
#             valid_len = min(PROBE_SIZE, grad_flat.numel())
#             grad_probe = grad_flat[:valid_len].float()
            
#             # Metric 1: Energy (Accumulate Norms)
#             fo_energy[layer_id] += grad_probe.norm(2).item()
            
#             # Metric 2: Displacement (Accumulate Vectors)
#             fo_disp_vec[layer_id][:valid_len] += grad_probe
            
#         # 清空梯度，为 ZO 做准备 (尽管 MeZO 不需要 .grad，但保持清洁)
#         model.zero_grad()
        
#         # =================================================
#         # Part B: Real Zeroth-Order (MeZO) Estimation
#         # =================================================
#         # 1. 调用我们重写的 MeZO 逻辑，获取真实的投影梯度标量
#         # 这会执行两次 Forward Pass (L+, L-)
#         zo_projected_grad, step_seed = analysis_trainer.get_zo_grad_info(model, inputs)
        
#         # 2. 记录 ZO 数据
#         # 我们需要根据 seed 重现 z，以计算更新向量 update = projected_grad * z
#         torch.manual_seed(step_seed)
        
#         for name, param in analysis_trainer.named_parameters_to_optim:
#             layer_id = get_layer_id(name)
            
#             # 生成 z (Standard Gaussian)
#             # 必须与 MeZO 内部生成逻辑完全一致
#             z = analysis_trainer.generate_random_noise(
#                 param.data.size(), 
#                 param.data.device, 
#                 param.data.dtype, 
#                 args.perturb_type # 通常是 'Gaussian'
#             )
            
#             # 计算 ZO 估计的梯度/更新量
#             # MeZO Update Rule: theta = theta - lr * (projected_grad * z)
#             # 这里我们记录 "projected_grad * z" 作为梯度的估计量
#             zo_grad_estimate = zo_projected_grad * z
            
#             # 取出 Probe 部分
#             zo_flat = zo_grad_estimate.view(-1)
#             valid_len = min(PROBE_SIZE, zo_flat.numel())
#             zo_probe = zo_flat[:valid_len].float()
            
#             # Metric 1: Energy (Accumulate Norms)
#             zo_energy[layer_id] += zo_probe.norm(2).item()
            
#             # Metric 2: Displacement (Accumulate Vectors)
#             zo_disp_vec[layer_id][:valid_len] += zo_probe

#     # --- 5. 结果汇总与计算 ---
#     valid_layers = sorted([k for k in fo_energy.keys() if 0 <= k < 9999])
    
#     results = {
#         "layer_ids": valid_layers,
        
#         # Energy (Path Length)
#         "fo_energy_raw": [],
#         "zo_energy_raw": [],
        
#         # Net Displacement (Vector Sum Norm)
#         "fo_displacement_norm": [],
#         "zo_displacement_norm": [],
#     }
    
#     for lid in valid_layers:
#         results["fo_energy_raw"].append(fo_energy[lid])
#         results["zo_energy_raw"].append(zo_energy[lid])
        
#         # 计算累积向量的模长
#         results["fo_displacement_norm"].append(fo_disp_vec[lid].norm(2).item())
#         results["zo_displacement_norm"].append(zo_disp_vec[lid].norm(2).item())

#     # --- 归一化 (为了绘图对比形状) ---
#     def normalize_list(l):
#         arr = np.array(l)
#         if arr.max() == 0: return l
#         return (arr / arr.max()).tolist()
    
#     results["fo_energy_normalized"] = normalize_list(results["fo_energy_raw"])
#     results["zo_energy_normalized"] = normalize_list(results["zo_energy_raw"])
    
#     results["fo_displacement_normalized"] = normalize_list(results["fo_displacement_norm"])
#     results["zo_displacement_normalized"] = normalize_list(results["zo_displacement_norm"])

#     # 保存文件
#     output_path = os.path.join(args.output_dir, "real_mezo_analysis.json")
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     with open(output_path, "w") as f:
#         json.dump(results, f, indent=4)
        
#     print(f"\nAnalysis Complete. Data saved to {output_path}")

#     # ==========================================
#     # 新增：即时计算并打印相关性 (Evidence of Blindness)
#     # ==========================================
#     from scipy.stats import pearsonr
    
#     # 1. Energy Correlation (资源分配相关性)
#     # 预期：极低 (接近 0)，因为 MeZO 是平的，Adam 是 U 型的
#     r_energy, _ = pearsonr(results["fo_energy_normalized"], results["zo_energy_normalized"])
    
#     # 2. Displacement Correlation (有效信号相关性)
#     # 预期：较低 (< 0.5)，证明累积未能有效去噪
#     r_disp, _ = pearsonr(results["fo_displacement_normalized"], results["zo_displacement_normalized"])
    
#     print("\n" + "="*40)
#     print("      SCIENTIFIC PROOF OF BLINDNESS      ")
#     print("="*40)
#     print(f"Steps Accumulated: {num_steps}")
#     print("-" * 40)
#     print(f"Energy Correlation (Resource Allocation):")
#     print(f"  r = {r_energy:.4f}")
#     print(f"  Interpretation: {'BLIND (Uncorrelated)' if r_energy < 0.5 else 'ALIGNED'}")
#     print("-" * 40)
#     print(f"Displacement Correlation (Effective Signal):")
#     print(f"  r = {r_disp:.4f}")
#     print(f"  Interpretation: {'NOISE DOMINATED' if r_disp < 0.5 else 'SIGNAL REVEALED'}")
#     print("="*40)
#     print("Note: A low 'r' (<0.3) proves that MeZO fails to match")
#     print("the layer-wise sensitivity structure of Adam.")

# if __name__ == "__main__":
#     main()






import os
import sys
import torch
import json
import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import pearsonr
from transformers import (
    HfArgumentParser, 
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed,
    DataCollatorForTokenClassification
)

sys.path.append(os.getcwd())
from arguments import LMZOTrainingArguments
from tasks import get_task
from utils import process_dataset
from zo_core.trainer.mezo_trainer import MeZOTrainer

# =============================================================================
# Helper Class
# =============================================================================
class AnalysisMeZOTrainer(MeZOTrainer):
    def get_zo_grad_info(self, model, inputs):
        model.eval()
        self.zo_random_seed = np.random.randint(1000000000)
        
        # Forward +
        self._perturb_mezo(scaling_factor=1)
        # model(**inputs) handles device movement internally for model parallel
        loss1 = self.zo_forward(model, inputs)
        
        # Forward -
        self._perturb_mezo(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        # Restore
        self._perturb_mezo(scaling_factor=1)
        
        return projected_grad, self.zo_random_seed

# =============================================================================
# Main Logic
# =============================================================================

def get_layer_id(name):
    match = re.search(r"\.(layers|block|h|blocks)\.(\d+)\.", name)
    if match: return int(match.group(2))
    if "embed" in name: return -1 
    return 9999 

def main():
    parser = HfArgumentParser((LMZOTrainingArguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)
    args.trainer = "mezo"
    if not hasattr(args, 'zo_eps') or args.zo_eps is None:
        args.zo_eps = 1e-3
    
    # 1. Load Model
    print(f"Loading model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, config=config, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True
    )
    
    # 2. Data
    task = get_task(args.task_name)
    train_samples = task.sample_train_sets(num_train=64, num_dev=1, num_eval=1, num_train_sets=1, seed=42)[0]
    train_dataset = process_dataset(args, task, train_samples, tokenizer, is_training=True)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, collate_fn=collator, shuffle=True)

    # 3. Trainer Setup
    analysis_trainer = AnalysisMeZOTrainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=None,
        tokenizer=tokenizer, data_collator=collator
    )
    analysis_trainer.named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            analysis_trainer.named_parameters_to_optim.append((name, param))

    # 4. Probe Setup
    PROBE_SIZE = 2048
    
    # 预先识别所有有效的层ID
    valid_layer_ids = set()
    for name, param in model.named_parameters():
        lid = get_layer_id(name)
        if 0 <= lid < 9999:
            valid_layer_ids.add(lid)
    sorted_layer_ids = sorted(list(valid_layer_ids))
    print(f"Tracking layers: {sorted_layer_ids}")

    # Accumulators
    # 【修复点1】不再使用 lambda 初始化，而是在循环中动态分配设备
    fo_energy = defaultdict(float)
    zo_energy = defaultdict(float)
    fo_disp_vec = {} 
    zo_disp_vec = {}

    # History Recorder
    history = {
        "steps": [],
        "energy_corr": [],       
        "displacement_corr": []  
    }

    num_steps = 1000
    print(f"Running Analysis for {num_steps} steps...")
    
    model.train()
    iterator = iter(dataloader)
    
    # 获取主设备 (用于放置 input)
    # 对于 device_map='auto'，model.device 通常是第一层所在的设备
    main_device = model.device 
    
    for step in tqdm(range(num_steps), desc="Analyzing"):
        try:
            inputs = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            inputs = next(iterator)
        
        # 将输入移到主设备，HuggingFace 的 forward 会自动处理后续的跨卡传输
        inputs = {k: v.to(main_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # --- Part A: FO Ground Truth ---
        model.zero_grad()
        outputs = model(**inputs)
        outputs.loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is None: continue
            layer_id = get_layer_id(name)
            if layer_id not in valid_layer_ids: continue
            
            grad_flat = param.grad.data.view(-1)
            valid_len = min(PROBE_SIZE, grad_flat.numel())
            grad_probe = grad_flat[:valid_len].float()
            
            # 【修复点2】动态初始化在同一设备上
            if layer_id not in fo_disp_vec:
                fo_disp_vec[layer_id] = torch.zeros(PROBE_SIZE, device=param.device, dtype=torch.float32)
            
            fo_energy[layer_id] += grad_probe.norm(2).item()
            fo_disp_vec[layer_id][:valid_len] += grad_probe
            
        model.zero_grad()
        
        # --- Part B: ZO Estimation ---
        zo_scalar, seed = analysis_trainer.get_zo_grad_info(model, inputs)
        torch.manual_seed(seed)
        
        for name, param in analysis_trainer.named_parameters_to_optim:
            layer_id = get_layer_id(name)
            if layer_id not in valid_layer_ids: continue
            
            # z 已经在 param.device 上生成
            z = analysis_trainer.generate_random_noise(
                param.data.size(), param.data.device, param.data.dtype, args.perturb_type
            )
            zo_grad = (zo_scalar * z).view(-1)
            valid_len = min(PROBE_SIZE, zo_grad.numel())
            zo_probe = zo_grad[:valid_len].float()
            
            # 【修复点3】动态初始化在同一设备上
            if layer_id not in zo_disp_vec:
                zo_disp_vec[layer_id] = torch.zeros(PROBE_SIZE, device=param.device, dtype=torch.float32)

            zo_energy[layer_id] += zo_probe.norm(2).item()
            zo_disp_vec[layer_id][:valid_len] += zo_probe

        # --- Part C: Record Dynamic Correlation (Every 10 steps) ---
        if (step + 1) % 10 == 0:
            # item() 会把 tensor 转回 CPU float，所以这里列表生成式是安全的，不会有设备冲突
            vec_fo_energy = [fo_energy[lid] for lid in sorted_layer_ids]
            vec_zo_energy = [zo_energy[lid] for lid in sorted_layer_ids]
            
            vec_fo_disp = [fo_disp_vec[lid].norm(2).item() for lid in sorted_layer_ids]
            vec_zo_disp = [zo_disp_vec[lid].norm(2).item() for lid in sorted_layer_ids]
            
            if np.std(vec_fo_energy) > 1e-9 and np.std(vec_zo_energy) > 1e-9:
                r_en, _ = pearsonr(vec_fo_energy, vec_zo_energy)
            else:
                r_en = 0.0
                
            if np.std(vec_fo_disp) > 1e-9 and np.std(vec_zo_disp) > 1e-9:
                r_disp, _ = pearsonr(vec_fo_disp, vec_zo_disp)
            else:
                r_disp = 0.0
            
            history["steps"].append(step + 1)
            history["energy_corr"].append(r_en)
            history["displacement_corr"].append(r_disp)

    # --- 5. Save Results ---
    results = {
        "layer_ids": sorted_layer_ids,
        "fo_energy_normalized": [],
        "zo_energy_normalized": [],
        "fo_displacement_normalized": [],
        "zo_displacement_normalized": [],
        "history": history
    }
    
    # Fill Snapshot Data
    raw_fo_en = [fo_energy[lid] for lid in sorted_layer_ids]
    raw_zo_en = [zo_energy[lid] for lid in sorted_layer_ids]
    raw_fo_disp = [fo_disp_vec[lid].norm(2).item() for lid in sorted_layer_ids]
    raw_zo_disp = [zo_disp_vec[lid].norm(2).item() for lid in sorted_layer_ids]
    
    def normalize(l):
        arr = np.array(l)
        return (arr / (arr.max() + 1e-9)).tolist()

    results["fo_energy_normalized"] = normalize(raw_fo_en)
    results["zo_energy_normalized"] = normalize(raw_zo_en)
    results["fo_displacement_normalized"] = normalize(raw_fo_disp)
    results["zo_displacement_normalized"] = normalize(raw_zo_disp)

    output_path = os.path.join(args.output_dir, "real_mezo_analysis.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved analysis to {output_path}")
    print(f"Final Energy Corr: {history['energy_corr'][-1]:.4f}")
    print(f"Final Disp Corr: {history['displacement_corr'][-1]:.4f}")

if __name__ == "__main__":
    main()