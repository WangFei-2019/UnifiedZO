import torch
import numpy as np
import math
import re
import os
import json
from collections import Counter
from .base_zo_trainer import BaseZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaLeZOTrainer(BaseZOTrainer):
    """
    Implementation of AdaLeZO (Adaptive Layer-wise Zeroth-Order Optimization) with REPLACEMENT sampling.
    
    Mechanism:
    1. Divides parameters into groups (layers).
    2. Uses Multi-Armed Bandit logic (EMA rewards) to select a sparse subset of layers.
       - Sampling is WITH REPLACEMENT (a layer can be chosen multiple times).
    3. Estimates gradients using ZO only on active layers.
    4. Updates parameters using Inverse Probability Weighting (IPW), scaled by selection counts.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)
        
        # Initialize AdaLeZO state (Layer grouping and Bandit statistics)
        self.init_adalezo_state(model)
        
        # Track current active layers for the step
        self.current_active_layers = []
        self.current_layer_probs_map = {}
        self.current_layer_counts_map = {}
        self.num_active_draws = 0
        
        # Initial sampling of layers
        self._resample_layers()

    def init_adalezo_state(self, model):
        """
        Parses model parameters to group them by layers and initializes Bandit statistics.
        """
        self.params_by_layer = {}
        
        for name, param in model.named_parameters():
            if not param.requires_grad: 
                continue
            
            # Regex to extract layer index. Supports standard HF models (Llama, OPT, BERT, etc.)
            # Matches patterns like 'layers.0.', 'blocks.1.', 'h.2.'
            match = re.search(r"\.(layers|block|h|blocks)\.(\d+)\.", name)
            
            # Key determination:
            # - Layers: index from regex
            # - Embeddings: -1
            # - Head/Norms: 9999 (Generic high index)
            key = int(match.group(2)) if match else (-1 if "embed" in name else 9999)
            
            if key not in self.params_by_layer: 
                self.params_by_layer[key] = []
            self.params_by_layer[key].append((name, param))
            
        self.sorted_layer_keys = sorted(self.params_by_layer.keys())
        self.num_layers = len(self.sorted_layer_keys)
        
        logger.info(f"[AdaLeZO] Initialized. Total Optimization Groups (Layers): {self.num_layers}")

        # --- Bandit Statistics ---
        # N_i: Selection counts (kept for logging/analysis)
        self.layer_counts = torch.zeros(self.num_layers, device=self.args.device)
        # Q_i: Average reward (gradient magnitude)
        self.layer_avg_rewards = torch.zeros(self.num_layers, device=self.args.device)
        
        # Initialize Probabilities
        if self.args.adalezo_warm_start:
            # Warm start: Bias towards deeper layers (last 40%)
            self.layer_scores = torch.zeros(self.num_layers, device=self.args.device)
            with torch.no_grad():
                for i, key in enumerate(self.sorted_layer_keys):
                    if key > self.num_layers * 0.6:
                        self.layer_scores[i] = 1.0
            self.layer_probs = torch.nn.functional.softmax(self.layer_scores / self.args.adalezo_tau, dim=0)
        else:
            # Uniform initialization
            self.layer_probs = torch.ones(self.num_layers, device=self.args.device) / self.num_layers

        # Adaptive Scaling State (RMSProp-like)
        if self.args.adalezo_layer_momentum:
            self.layer_sq_grads = torch.zeros(self.num_layers, device=self.args.device) 

    def _resample_layers(self):
        """
        Selects active layers using WITH REPLACEMENT sampling.
        """
        # Number of samples to draw (k)
        k = max(1, int(self.args.adalezo_k_ratio * self.num_layers))
        self.num_active_draws = k
        
        # Normalize rewards for numerical stability
        max_reward = self.layer_avg_rewards.max()
        # Use rewards directly as scores (Exploitation)
        scores = self.layer_avg_rewards / max_reward if max_reward > 1e-6 else self.layer_avg_rewards
        
        # Convert scores to probabilities via Softmax (Temperature controlled)
        self.layer_probs = torch.nn.functional.softmax(scores / self.args.adalezo_tau, dim=0)

        # Mix with uniform distribution to ensure exploration (Gamma Mixing)
        # Critical for preventing starvation of low-probability layers
        self.layer_probs = (1 - self.args.adalezo_gamma) * self.layer_probs + self.args.adalezo_gamma * (1.0 / self.num_layers)

        # Sample k indices WITH REPLACEMENT
        active_indices = torch.multinomial(self.layer_probs, k, replacement=True)
        
        # Map indices to layer keys
        active_keys_raw = [self.sorted_layer_keys[i] for i in active_indices.tolist()]
        
        # Count occurrences of each layer (How many times each was selected)
        self.current_layer_counts_map = Counter(active_keys_raw)
        
        # Active layers are the UNIQUE keys selected
        self.current_active_layers = sorted(self.current_layer_counts_map.keys())

        # Store Probabilities for IPW calculation later (only for unique active layers)
        self.current_layer_probs_map = {
            key: self.layer_probs[self.sorted_layer_keys.index(key)].item() 
            for key in self.current_active_layers
        }

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        AdaLeZO Training Step:
        1. Perturb ONLY active layers (+).
        2. Forward.
        3. Perturb ONLY active layers (-).
        4. Forward.
        5. Update parameters.
        """
        model.eval()

        # 1. Sample Seed
        self.zo_random_seed = np.random.randint(1000000000)
        
        # 2. Perturb Active Layers (+)
        # Note: Even if a layer was selected 5 times, we only perturb it ONCE here.
        # The '5 times' logic is handled in the update step (gradient scaling).
        self._perturb_active_layers(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        # 3. Perturb Active Layers (-) : Move from +1 to -1 state (subtract 2)
        self._perturb_active_layers(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        # 4. Calculate Projected Gradient Estimate (Scalar proxy)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        # Restore Active Layers to original state (add 1)
        self._perturb_active_layers(scaling_factor=1)
        
        # --- [新增] 实验一代码插入点：Oracle 梯度相关性分析 ---
        # 建议每隔 N 步 (如 100 步) 运行一次，因为反向传播非常耗时
        if self.state.global_step % 100 == 0:
            self._analyze_gradient_correlation(model, inputs)
        # -----------------------------------------------------

        # 5. Update Parameters & Bandit Stats
        self._update_adalezo()
        
        # 6. Periodic Re-sampling (Stickiness)
        if self.state.global_step % self.args.adalezo_interval == 0:
            self._resample_layers()
            # Optional: Log probs
            self._log_probs()

        return loss1

    def _perturb_active_layers(self, scaling_factor):
        """
        Apply perturbations only to the layers selected by the Bandit.
        """
        # Deterministic seeding per layer to ensure consistency
        base_seed = self.zo_random_seed
        
        for layer_key in self.current_active_layers:
            # Seed depends on layer key to be unique but reproducible
            torch.manual_seed(base_seed + layer_key)
            
            for name, param in self.params_by_layer[layer_key]:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                param.data += scaling_factor * z * self.args.zo_eps

    def _update_adalezo(self):
        """
        Update parameters using IPW, scaled by selection counts.
        """
        args = self.args
        lr = self._get_learning_rate()
        
        step_reward = abs(self.projected_grad)

        # Iterate only over UNIQUE active layers
        for layer_key in self.current_active_layers:
            prob = self.current_layer_probs_map[layer_key]
            count = self.current_layer_counts_map[layer_key] # Number of times selected
            
            # --- IPW Calculation (With Replacement) ---
            # Unbiased Estimator Formula: 
            # Update = Sum_{j=1 to k} [ I(layer selected at j) * (g / (k * p)) ]
            #        = count * (g / (k * p))
            
            # Denominator uses Total Draws (k), not unique count
            raw_ipw = 1.0 / (prob * self.num_active_draws + 1e-8)
            ipw_weight = min(raw_ipw, args.adalezo_ipw_clip)
            
            # Scale factor includes the count (multiplicity)
            scale_factor = ipw_weight * count

            # --- Optional: Layer-wise Momentum (Adaptive Scaling) ---
            if args.adalezo_layer_momentum:
                idx = self.sorted_layer_keys.index(layer_key)

                # Estimate variance. 
                # Note: We update momentum stats based on "one observation" of the gradient energy.
                current_energy = self.projected_grad ** 2 
                
                self.layer_sq_grads[idx] = args.adalezo_beta * self.layer_sq_grads[idx] + (1 - args.adalezo_beta) * current_energy
                denom = torch.sqrt(self.layer_sq_grads[idx]) + 1e-8
                
                scale_factor = (scale_factor / denom).item()

            # --- Parameter Update ---
            torch.manual_seed(self.zo_random_seed + layer_key)
            for name, param in self.params_by_layer[layer_key]:
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                
                # Update term = g * z * IPW * Count
                update_term = self.projected_grad * z * scale_factor
                
                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                         param.data.add_(update_term + args.weight_decay * param.data, alpha=-lr)
                    else:
                         param.data.add_(update_term, alpha=-lr)
                else:
                    param.data.add_(update_term, alpha=-lr)

            # --- Update Bandit Stats ---
            # We update the stats ONCE per step per layer, regardless of 'count'.
            # Reason: The gradient signal 'step_reward' is observed only once this step.
            idx = self.sorted_layer_keys.index(layer_key)
            self.layer_counts[idx] += count # Track total allocated budget
            
            # EMA Update
            self.layer_avg_rewards[idx] = (1 - args.adalezo_ema_alpha) * self.layer_avg_rewards[idx] + args.adalezo_ema_alpha * step_reward

    def _log_probs(self):
        """
        Helper to log probability evolution to file.
        """
        if self.layer_probs is None: return
        
        data = {
            "step": self.state.global_step,
            "probs": self.layer_probs.detach().cpu().tolist(),
            "active": self.current_active_layers,
            "counts": dict(self.current_layer_counts_map), # Log counts for debug
            "normalized_rewards": self.layer_avg_rewards.detach().cpu().tolist(), 
        }
        
        # Save to output dir
        save_path = os.path.join(self.args.output_dir, "adalezo_probs.jsonl")
        with open(save_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _analyze_gradient_correlation(self, model, inputs):
        """
        实验一辅助函数：计算 ZO 估计梯度与真实梯度（Oracle）的余弦相似度并保存。
        修正版：增加对 tuple 输出的处理。
        """
        import torch.nn.functional as F
        import os
        import json

        # 1. 计算真实梯度 (True Gradient)
        model.zero_grad()
        
        # 确保 inputs 在正确的设备上 (如果之前未处理)
        # inputs = self._prepare_inputs(inputs) # 视情况取消注释，通常 training_step 里的 inputs 已经处理好了
        
        with torch.enable_grad():
            outputs = model(**inputs)
            
            # --- [修复] 兼容 Tuple 和 ModelOutput ---
            if hasattr(outputs, "loss"):
                loss_true = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs:
                loss_true = outputs["loss"]
            else:
                # 如果是 tuple，通常第一个元素是 loss
                loss_true = outputs[0]
            # -------------------------------------

            # 检查 loss 是否为 None (有时 inference 模式下某些模型不返回 loss)
            if loss_true is None:
                # 如果 inputs 里没有 labels，模型可能不计算 loss。确保 inputs 包含 labels。
                logger.warning("[Analysis] True gradient calculation skipped: Model did not return a loss.")
                model.zero_grad()
                return

            loss_true.backward()
        
        # 2. 准备向量容器
        zo_vec_list = []
        true_grad_vec_list = []
        
        # 3. 遍历当前活跃层，构建 ZO 更新向量和真实梯度向量
        args = self.args
        
        # 预计算一些不需要梯度的参数以复现 _update_adalezo 的逻辑
        # 注意：这里需要重新获取当前的 ipw 和 counts，因为它们在 _resample_layers 后是固定的
        
        for layer_key in self.current_active_layers:
            # 获取该层参数
            layer_params = self.params_by_layer[layer_key]
            
            # 3.1 恢复该层的 ZO 配置
            prob = self.current_layer_probs_map[layer_key]
            count = self.current_layer_counts_map[layer_key]
            
            # 复现 IPW 和 Scale Factor 逻辑 (必须与 _update_adalezo 严格一致)
            current_raw_ipw = 1.0 / (prob * self.num_active_draws + 1e-8)
            ipw_weight = min(current_raw_ipw, args.adalezo_ipw_clip)
            base_scale = ipw_weight * count
            
            # 复现 RMS (Adaptive Scaling) 逻辑
            idx = self.sorted_layer_keys.index(layer_key)
            if args.adalezo_layer_momentum:
                layer_grad_est = self.projected_grad * ipw_weight
                current_energy = layer_grad_est ** 2
                # 模拟更新后的二阶矩 (窥视未来一步的 denom)
                simulated_sq_grad = args.adalezo_beta * self.layer_sq_grads[idx] + (1 - args.adalezo_beta) * current_energy
                # 使用您代码中实际的 epsilon
                denom = torch.sqrt(simulated_sq_grad) + 1e-8
                base_scale = base_scale / denom.item()
            
            # 3.2 重建该层的 ZO 方向向量 (z)
            # 关键：必须使用与 perturb 和 update 相同的 Seed
            torch.manual_seed(self.zo_random_seed + layer_key)
            
            for name, param in layer_params:
                # 如果真实梯度不存在 (例如凍結参数)，跳过
                if param.grad is None: 
                    # 消耗随机数以保持 RNG 同步
                    self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                    continue
                
                # 生成噪声 z
                z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, 'Gaussian')
                
                # ZO 估计的梯度方向 (Effective Update Direction)
                zo_grad_component = self.projected_grad * z * base_scale
                
                # 收集向量 (Flatten)
                zo_vec_list.append(zo_grad_component.flatten().float().cpu())
                true_grad_vec_list.append(param.grad.flatten().float().cpu())

        # 4. 如果没有收集到梯度（例如所有层都被冻结），直接返回
        if not zo_vec_list:
            model.zero_grad()
            return

        # 5. 拼接成大向量并计算 Cosine Similarity
        zo_full_vec = torch.cat(zo_vec_list)
        true_full_vec = torch.cat(true_grad_vec_list)
        
        # 避免除以零
        if true_full_vec.norm() < 1e-8 or zo_full_vec.norm() < 1e-8:
             cos_sim = 0.0
        else:
             cos_sim = F.cosine_similarity(zo_full_vec.unsqueeze(0), true_full_vec.unsqueeze(0)).item()
        
        zo_norm = torch.norm(zo_full_vec).item()
        true_norm = torch.norm(true_full_vec).item()

        # 6. 清理梯度，防止影响后续 ZO 步骤
        model.zero_grad()

        # 7. 写入文件
        log_data = {
            "step": self.state.global_step,
            "projected_grad": self.projected_grad,
            "cosine_similarity": cos_sim,
            "zo_norm": zo_norm,
            "true_grad_norm": true_norm,
            "ratio_zo_true": zo_norm / (true_norm + 1e-8)
        }
        
        # # 确保输出目录存在
        # if not os.path.exists(self.args.output_dir):
        #     os.makedirs(self.args.output_dir, exist_ok=True)

        save_path = os.path.join("/workspace/wangfei154/project/UnifiedZO/large_models/result/analysis", "analysis_gradient_correlation.jsonl")
        with open(save_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")
            
        logger.info(f"[Analysis] Step {self.state.global_step}: CosSim={cos_sim:.6f}, ZO_Norm={zo_norm:.4f}, True_Norm={true_norm:.4f}")