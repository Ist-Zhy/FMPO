import os
import argparse
import yaml
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler  # [新增] 混合精度
from peft import LoraConfig, get_peft_model      # [新增] LoRA 支持

# --- 项目模块 ---
from Model.Flow_MMIF import FlowMatching, UnetVelocity
from Reward.fusion_reward_plusplus import FusionReward 
from Dataset.Load_Data import Restoration 
import warnings
warnings.filterwarnings("ignore")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def cycle(dl):
    while True:
        for data in dl:
            yield data

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Consistency-Aware GRPO RL (Multi-step Theory)")
    parser.add_argument('--config', type=str, default='configs/Flow_Train_RL_plus.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed_everything(args.seed)
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    diffusion_params = config['Diffusion']
    dataset_opt = config['Dataset_Configs']
    rl_params = config['RL_parameters']

    save_dir = rl_params['results_folder']
    os.makedirs(save_dir, exist_ok=True)

    # ====================================================
    # 1. 初始化模型
    # ====================================================
    print("Initializing Model...")
    unet = UnetVelocity(
        dim=diffusion_params.get('dim', 64),
        dim_mults=(1, 1, 1, 1),
        condition=diffusion_params['condition']
    ).to(device)

    flow_model = FlowMatching(
        unet,
        image_size=diffusion_params['image_size'],
        timesteps=1000,
        sampling_timesteps=diffusion_params['sampling_timesteps'],
        loss_type=diffusion_params['loss_type'],
        condition=diffusion_params['condition'],
        sigma_min=0.0,
        sigma_max=0.0
    ).to(device)

    # ====================================================
    # 2. 加载 SFT 预训练权重
    # ====================================================
    sft_path = rl_params['sft_checkpoint_path']
    if os.path.exists(sft_path):
        print(f"Loading SFT weights from {sft_path}...")
        checkpoint = torch.load(sft_path, map_location=device)

        if 'ema' in checkpoint:
            source_state_dict = checkpoint['ema']
        elif 'model' in checkpoint:
            source_state_dict = checkpoint['model']
        else:
            source_state_dict = checkpoint

        target_keys = list(flow_model.state_dict().keys())
        new_state_dict = {}
        for k, v in source_state_dict.items():
            new_k = k.replace('online_model.', '').replace('module.', '')
            if new_k in target_keys:
                new_state_dict[new_k] = v
        try:
            flow_model.load_state_dict(new_state_dict, strict=True)
            print(">> SFT Weights Loaded Successfully!")
        except Exception as e:
            raise RuntimeError("SFT checkpoint does not match the Stage II architecture and width") from e
    else:
        print(f">> Error: SFT Checkpoint not found at {sft_path}")
        exit(1)

    # ====================================================
    # 3. 注入 LoRA (替代手动冻结，支持多步理论微调)
    # ====================================================
    print("Injecting LoRA Adapters for Consistency Fine-tuning...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        # [修改这里] 去掉 "to_out"，只保留具体的层名称
        target_modules=["to_qkv", "to_out.0"], 
        lora_dropout=0.05, 
        bias="none"
    )
    # 包装模型
    flow_model.model = get_peft_model(flow_model.model, lora_config)
    flow_model.model.print_trainable_parameters()

    # 优化器
    optimizer = AdamW(flow_model.parameters(), lr=float(rl_params['lr']), weight_decay=1e-6)
    
    # 混合精度 Scaler
    scaler = GradScaler()

    # ====================================================
    # 4. 初始化组件
    # ====================================================
    reward_fn = FusionReward(
        device=device,
        use_clip_reward=True,
        dinov3_strategy="select_3_6_9_11"
    )

    if not reward_fn.dinov3.has_model or not reward_fn.clip or not reward_fn.clip.enabled:
        raise RuntimeError("DINOv3 and OpenCLIP rewards must load successfully; check FMPO_DINOV3_REPO, FMPO_DINOV3_WEIGHTS, and FMPO_CLIP_WEIGHTS.")

    print("Initializing DataLoader...")
    data_manager = Restoration(config=dataset_opt['IVF'])
    train_loader, _ = data_manager.get_loaders()
    data_iter = cycle(train_loader)

    def get_batch():
        batch = next(data_iter)
        # 范围 [0, 1] -> [-1, 1]
        ir = batch['ir'].to(device) * 2.0 - 1.0
        vi = batch['vi'].to(device) * 2.0 - 1.0
        return ir, vi

    def get_epsilon(step, total_steps, start=0.15, end=0.05):
        progress = float(step) / float(max(1, total_steps))
        return start + (end - start) * progress

    # ====================================================
    # 5. Consistency-Aware GRPO 训练循环
    # ====================================================
    group_size = rl_params.get('group_size', 4)
    total_steps = int(rl_params['train_num_steps'])
    
    # [理论参数] 扰动尺度：模拟轨迹偏离 Source 的程度
    # 这个值不用太大，0.1~0.2 足够让模型学会鲁棒性，同时不至于太难学
    perturb_scale = 0.1 

    print(f"Starting Consistency-Aware Training ({total_steps} steps)...")

    flow_model.train()
    # 确保 BN 层即使在 LoRA 下也保持 eval 模式
    for module in flow_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    pbar = tqdm(range(1, total_steps + 1))

    for step in pbar:
        current_eps = get_epsilon(step, total_steps)

        # ---------------- 1. 准备数据 ----------------
        x_ir, x_vi = get_batch()
        batch_size = x_ir.shape[0]
        x_source = (x_ir + x_vi) / 2.0  # 起点 x_0

        # ---------------- 2. [核心] 构造多步轨迹数据 (x_t, t) ----------------
        # A. 随机采样时间步 t ~ U[0, 1] (避免 t=1 导致的数值问题，乘 0.99)
        t = torch.rand(batch_size, device=device) * 0.99 
        
        # B. 构造中间态 x_t
        # 理论：OT 轨迹是线性的，我们假设终点在 x_source 附近的一个邻域内
        # x_t = (1-t) * x_source + t * (x_source + noise)
        # 简化后 -> x_t = x_source + t * noise
        # 这样随着 t 增大，输入离 source 越远，模型需要预测的 v 越能把图拉回来
        noise_sim = torch.randn_like(x_source) * perturb_scale
        x_t_input = x_source + t.view(-1, 1, 1, 1) * noise_sim

        # C. 构造条件输入
        if flow_model.condition:
            x_in = torch.cat((x_t_input, x_ir, x_vi), dim=1)
        else:
            x_in = x_t_input

        # ---------------- 3. 混合精度前向与采样 ----------------
        with autocast():
            # [策略前向] 预测从当前 x_t 出发的速度场 v
            v_base_curr = flow_model.model(x_in, t) 
            
            # [一致性投影] 即使是训练中，我们也用 ODE 线性性质推断 x_1
            # 公式: x_1 = x_t + (1-t) * v
            dt = (1.0 - t).view(-1, 1, 1, 1)
            
            # --- 探索 (Exploration) ---
            v_base_detached = v_base_curr.detach()
            
            # 扩展 batch 以进行 Group 采样
            v_base_expanded = v_base_detached.repeat_interleave(group_size, dim=0)
            
            # Action Space Noise (在速度场 v 上加噪声)
            noise_action = torch.randn_like(v_base_expanded) * current_eps
            v_samples = v_base_expanded + noise_action 

            # 计算对应的 Sampled x_1 (Projected Terminal State)
            # 需要扩展 x_t 和 dt 以匹配 group_size
            x_t_expanded = x_t_input.repeat_interleave(group_size, dim=0)
            dt_expanded = dt.repeat_interleave(group_size, dim=0)
            
            # 这里的 x_start_samples 就是我们假设只要按 v 走到底就会得到的最终图像
            x_start_samples = torch.clamp(x_t_expanded + dt_expanded * v_samples, -1, 1)

        # ---------------- 4. 计算奖励 (Reward) - 保持 FP32 精度 ----------------
        x_ir_flat = x_ir.repeat_interleave(group_size, dim=0)
        x_vi_flat = x_vi.repeat_interleave(group_size, dim=0)

        with torch.no_grad():
            r_flat, info_dict, _ = reward_fn.compute_reward(
                x_start_samples.float(), 
                x_ir_flat.float(), 
                x_vi_flat.float()
            )
            rewards = r_flat.view(batch_size, group_size)
            
            # Advantage Calculation
            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True).clamp(min=1e-4)
            advantages = (rewards - mean_r) / std_r

        # ---------------- 5. GRPO Loss 计算 (AMP) ----------------
        with autocast():
            # 目标: 让模型输出的 v_base_curr 接近那些能产生好结果的 v_samples
            target_v = v_samples.view(batch_size, group_size, *v_base_detached.shape[1:]).detach()
            v_base_curr_expanded = v_base_curr.unsqueeze(1) 

            # 权重处理
            weights = F.relu(advantages)
            weight_sum = weights.sum(dim=1, keepdim=True)
            norm_weights = weights / (weight_sum + 1e-6)
            norm_weights_exp = norm_weights.view(batch_size, group_size, 1, 1, 1)

            # Weighted MSE Loss
            squared_diff = (v_base_curr_expanded - target_v) ** 2
            loss = (squared_diff * norm_weights_exp).sum(dim=1).mean()

        # ---------------- 6. 反向传播 (AMP) ----------------
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
# ---------------- 7. 日志与保存 ----------------
        if step % 10 == 0:
            batch_avg_reward = rewards.mean().item()
            # 基础日志：L=Loss, R=总Reward
            log_str = f"L:{loss.item():.4f} | R:{batch_avg_reward:.3f}"
            
            # [修改] 遍历 info_dict 打印所有子奖励组件
            for k, v in info_dict.items():
                # 跳过 'Total' 键，因为 R 已经是 Total 了，避免日志太长重复
                if k == 'Total': continue 
                # 格式化打印，保留2位小数
                log_str += f" | {k}:{v:.2f}"
            
            pbar.set_description(f"Step {step} | {log_str}")

        if step % rl_params.get('save_every', 500) == 0:
            save_path = os.path.join(save_dir, f'rl_step_{step}.pt')
            torch.save({
                'model': flow_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, save_path)

    # 保存最终模型
    torch.save({'model': flow_model.state_dict()}, os.path.join(save_dir, 'rl_final.pt'))
    print(f"RL Training Finished! Model saved to {save_dir}")

if __name__ == '__main__':
    main()