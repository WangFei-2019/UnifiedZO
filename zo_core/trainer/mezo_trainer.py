import torch
import numpy as np
from .base_zo_trainer import BaseZOTrainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MeZOTrainer(BaseZOTrainer):
    """
    Implementation of MeZO (Memory-efficient Zeroth-Order Optimization).
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        MeZO Step:
        1. Sample z
        2. L1 = Loss(theta + z)
        3. L2 = Loss(theta - z)
        4. Grad ~ (L1 - L2) / (2 eps) * z
        5. Update theta
        """
        model.eval()

        # 1. Sample Seed
        self.zo_random_seed = np.random.randint(1000000000)
        
        # 2. Forward +
        self._perturb_mezo(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        # 3. Forward - (From +1 to -1 requires -2 step)
        self._perturb_mezo(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        # 4. Calculate Projected Gradient Estimate
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        
        # Restore parameters to original state (from -1 to 0 requires +1)
        self._perturb_mezo(scaling_factor=1)

        # --- [新增] 实验一代码插入点：Oracle 梯度相关性分析 ---
        # 建议每隔 N 步 (如 100 步) 运行一次，因为反向传播非常耗时
        if self.state.global_step % 100 == 0:
            self._analyze_gradient_correlation(model, inputs)
        # -----------------------------------------------------

        # 5. Update Parameters
        self._update_mezo()
        
        return loss1

    def _perturb_mezo(self, scaling_factor):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
            param.data += scaling_factor * z * self.args.zo_eps

    def _update_mezo(self):
        torch.manual_seed(self.zo_random_seed)
        lr = self._get_learning_rate()
        
        for name, param in self.named_parameters_to_optim:
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
            
            # Update rule: theta = theta - lr * (grad_est + weight_decay)
            update = self.projected_grad * z
            
            # Apply weight decay
            if self.args.weight_decay > 0 and "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                update += self.args.weight_decay * param.data

            param.data -= lr * update

    def _analyze_gradient_correlation(self, model, inputs):
        """
        [MeZO 版本] 實驗輔助函數：計算 MeZO 估計梯度與真實梯度（Oracle）的餘弦相似度並保存。
        """
        import torch.nn.functional as F
        import os
        import json

        # 1. 計算真實梯度 (True Gradient)
        # 必須顯式開啟梯度計算，因為 ZO 訓練通常在 no_grad 下運行
        model.zero_grad()
        
        with torch.enable_grad():
            outputs = model(**inputs)
            
            # --- 兼容 Tuple 和 ModelOutput 的 Loss 提取 ---
            if hasattr(outputs, "loss"):
                loss_true = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs:
                loss_true = outputs["loss"]
            else:
                # 如果是 tuple，通常第一個元素是 loss
                loss_true = outputs[0]
            # -------------------------------------------

            if loss_true is None:
                return

            loss_true.backward()
        
        # 2. 準備向量容器
        zo_vec_list = []
        true_grad_vec_list = []
        
        # 3. 遍歷所有參與優化的參數，重建 ZO 更新向量
        # MeZO 是全量更新，所以我們遍歷 named_parameters_to_optim
        
        # [關鍵] 必須重置隨機與 training_step 中產生 z 時一致
        torch.manual_seed(self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            # 如果真實梯度不存在 (例如凍結參數)，我們也要消耗隨機數以保持 RNG 同步
            if param.grad is None: 
                self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
                continue
            
            # A. 重現隨機噪聲 z
            # MeZO 默認使用 Normal (Gaussian) 或其他分佈，需傳入 self.args.perturb_type
            z = self.generate_random_noise(param.data.size(), param.data.device, param.data.dtype, self.args.perturb_type)
            
            # B. 重建 ZO 估計梯度 (Effective Gradient Estimate)
            # MeZO 的梯度估計方向 = projected_grad * z
            # 這裡我們不乘學習率，只看梯度的方向和相對大小
            zo_grad_component = self.projected_grad * z
            
            # C. 收集向量 (強制轉為 float32 防止 FP16 溢出)
            zo_vec_list.append(zo_grad_component.flatten().float().cpu())
            true_grad_vec_list.append(param.grad.flatten().float().cpu())

        # 4. 如果沒有收集到梯度，直接返回
        if not zo_vec_list:
            model.zero_grad()
            return

        # 5. 拼接並計算指標
        zo_full_vec = torch.cat(zo_vec_list)
        true_full_vec = torch.cat(true_grad_vec_list)
        
        # 防止除以零
        if true_full_vec.norm() < 1e-8 or zo_full_vec.norm() < 1e-8:
             cos_sim = 0.0
        else:
             cos_sim = F.cosine_similarity(zo_full_vec.unsqueeze(0), true_full_vec.unsqueeze(0)).item()
        
        zo_norm = torch.norm(zo_full_vec).item()
        true_norm = torch.norm(true_full_vec).item()

        # 6. 清理梯度
        model.zero_grad()

        # 7. 寫入日誌文件
        # 確保輸出目錄存在
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)

        log_data = {
            "step": self.state.global_step,
            "projected_grad": self.projected_grad,
            "cosine_similarity": cos_sim,
            "zo_norm": zo_norm,
            "true_grad_norm": true_norm,
            "ratio_zo_true": zo_norm / (true_norm + 1e-8)
        }
        
        save_path = os.path.join("/workspace/wangfei154/project/UnifiedZO/large_models/result/analysis", "analysis_gradient_correlation-mezo.jsonl")
        with open(save_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")
            
        logger.info(f"[MeZO Analysis] Step {self.state.global_step}: CosSim={cos_sim:.6f}, Ratio={log_data['ratio_zo_true']:.2f}")