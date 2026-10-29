import os
import torchvision.transforms as transforms
import random
# ===================== 配置区域 =====================
REPO_DIR = os.path.abspath(os.environ.get("FMPO_DINOV3_REPO", "external/dinov3"))
CHECKPOINT_PATH = os.environ.get("FMPO_DINOV3_WEIGHTS", "weights/dinov3_vitb16.pth")

# ===================== CLIP Setup =====================
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: open_clip_torch not installed. Run 'pip install open_clip_torch'")

# =================================================================
# 🦕 DINOv3 独立模块
# =================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# =================================================================
# 🦕 DINOv3 独立模块 (Refactored: Max Strategy)
# =================================================================
class DINOv3LocalReward(nn.Module):
    def __init__(self, device="cuda", strategy="select_3_6_9_11"):
        super().__init__()
        self.device = device
        self.patch_size = 16
        self.strategy = strategy
        self.has_model = False

        # 路径配置
        self.repo_dir = REPO_DIR
        self.ckpt_path = CHECKPOINT_PATH

        print(f"Loading DINOv3 (Strategy: {strategy}) via torch.hub (Local)...")
        if not os.path.exists(self.repo_dir) or not os.path.exists(self.ckpt_path):
            print(f"Error: DINOv3 path not found.")
            return

        try:
            self.model = torch.hub.load(
                repo_or_dir=self.repo_dir,
                model='dinov3_vitb16',
                source='local',
                weights=self.ckpt_path
            )
            self.model.to(device).eval()
            # 彻底冻结参数
            for p in self.model.parameters():
                p.requires_grad = False
            self.has_model = True
        except Exception as e:
            print(f"Error loading DINOv3: {e}")
            return

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.n_blocks = len(self.model.blocks) if hasattr(self.model, 'blocks') else 12

    def _preprocess(self, img):
        """ 预处理：Resize 为 patch 整数倍 + 归一化 """
        B, C, H, W = img.shape
        # 确保尺寸是 Patch 的倍数
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        # 避免尺寸过小
        new_H = max(new_H, self.patch_size * 4)
        new_W = max(new_W, self.patch_size * 4)

        if new_H != H or new_W != W:
            img = F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=False)

        # 灰度转RGB
        if C == 1:
            img = img.repeat(1, 3, 1, 1)

        return (img - self.mean) / self.std

    def get_all_layer_features(self, img):
        if not self.has_model: return None
        img_in = self._preprocess(img.to(self.device))

        with torch.no_grad():
            # 获取所有层的 Block Output (Patch Tokens)
            # 输出是一个 list，包含 n_blocks 个 tensor
            all_layers = self.model.get_intermediate_layers(
                img_in, n=self.n_blocks, reshape=False, return_class_token=False, norm=True
            )
        return all_layers

    def compute_semantic_from_features(self, feat_list_f, feat_list_i, feat_list_v, tau=0.3):
        """
        计算语义一致性奖励 (使用 Max Strategy)
        """
        if feat_list_f is None:
            return torch.zeros(1, device=self.device)

        # --- 内部函数: 层级特征聚合 ---
        def aggregate_layers(feat_list):
            total_depth = len(feat_list)
            # 解析策略 "select_3_6_9_11"
            if self.strategy.startswith('select_'):
                try:
                    indices = [int(x) for x in self.strategy.split('_')[1:]]
                    # 选取对应层
                    selected = [feat_list[min(max(0, i), total_depth - 1)] for i in indices]
                    # 每一层先做一次归一化，防止某一层数值过大主导
                    normed = [F.normalize(f, dim=-1) for f in selected]
                    # 在层级维度取平均，得到该图像的综合特征表达
                    return torch.stack(normed).mean(dim=0)
                except:
                    pass
            # 默认回退：取最后一层
            return F.normalize(feat_list[-1], dim=-1)

        # 1. 获得每个模态的综合特征 (B, N, C)
        raw_f = aggregate_layers(feat_list_f)
        raw_i = aggregate_layers(feat_list_i)
        raw_v = aggregate_layers(feat_list_v)

        # 2. 再次归一化 (确保在单位球面上比较方向)
        feat_f = F.normalize(raw_f, dim=-1)
        feat_i = F.normalize(raw_i, dim=-1)
        feat_v = F.normalize(raw_v, dim=-1)

        # 3. 处理 Batch Size 不匹配 (例如单张源图 vs 多张生成图)
        B = feat_f.shape[0]
        if feat_i.shape[0] == 1 and B > 1: feat_i = feat_i.expand(B, -1, -1)
        if feat_v.shape[0] == 1 and B > 1: feat_v = feat_v.expand(B, -1, -1)
        # =========================================================
        # 🔥 核心修改：Max Fusion Strategy
        # =========================================================
        # 我们构建一个“理想目标特征” target。
        # 逻辑：对于每一个特征维度，取 IR 和 VI 中绝对值响应更大的那个。
        # 这种方式能保留显著的热目标(IR)和清晰的纹理(VI)。
        target = torch.max(feat_i, feat_v)
        # 再次归一化 target，因为 max 操作可能会改变模长
        target = F.normalize(target, dim=-1)
        # 4. 计算 Cosine Similarity
        # feat_f 与 target 的相似度
        cos_sim = torch.sum(feat_f * target, dim=-1)  # (B, N)
        # 截断以防数值误差导致 NaN
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        # 5. 转换为角度距离损失 -> 奖励
        # 角度越小(相似度越高)，theta越小，exp(-theta)越大
        theta = torch.acos(cos_sim)
        reward_map = torch.exp(-theta / tau)
        # 对所有 Token 取平均，得到整张图的分数
        return reward_map.mean(dim=1)

# =================================================================
# 📎 CLIP 独立模块
# =================================================================
class CLIPLogicReward(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.enabled = CLIP_AVAILABLE

        # 指定您的本地权重路径
        self.model_path = os.environ.get('FMPO_CLIP_WEIGHTS', 'weights/open_clip_pytorch_model.bin')
        self.model_name = 'ViT-B-16'  # 对应您权重的模型架构

        if not self.enabled:
            return

        print(f"Loading OpenCLIP ({self.model_name}) from {self.model_path} for LDFusion Logic...")
        try:
            # 1. 使用 open_clip 加载模型
            # 注意: pretrained 指定为您的本地路径
            self.model, _, _ = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.model_path,
                device=device
            )
            # 2. 获取对应的 tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_name)

            self.model.eval()
            for p in self.model.parameters(): p.requires_grad = False

            # === LDFusion 核心 Prompt 设置 ===
            vi_text = ['a visible gray image']
            ir_text = ['an infrared image']
            self.target_prompt = ['a clear image with detailed background and salient objects']

            with torch.no_grad():
                # 3. 使用 self.tokenizer 进行编码
                vit_token = self.tokenizer(vi_text).to(device)
                self.vit_feature = self.model.encode_text(vit_token).float()
                # 归一化特征
                self.vit_feature = F.normalize(self.vit_feature, dim=-1)

                irt_token = self.tokenizer(ir_text).to(device)
                self.irt_feature = self.model.encode_text(irt_token).float()
                self.irt_feature = F.normalize(self.irt_feature, dim=-1)

                target_token = self.tokenizer(self.target_prompt).to(device)
                self.prompt_feature = self.model.encode_text(target_token).float()
                self.prompt_feature = F.normalize(self.prompt_feature, dim=-1)

                # 预计算文本方向向量 (Target - Source)
                self.delta_txt_vi = F.normalize(self.prompt_feature - self.vit_feature, dim=-1)
                self.delta_txt_ir = F.normalize(self.prompt_feature - self.irt_feature, dim=-1)

            # 注册预处理所需的均值和方差 (OpenCLIP 默认值)
            self.register_buffer('mean',
                                 torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1))
            self.register_buffer('std',
                                 torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1))

        except Exception as e:
            print(f"CLIP Load Error: {e}")
            self.enabled = False

    def _preprocess(self, img):
        # 1. 调整大小到 224x224 (ViT 标准输入)
        if img.shape[-1] != 224 or img.shape[-2] != 224:
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

        # 2. 灰度转 3 通道
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)

        # 3. 标准化
        return (img - self.mean) / self.std

    def compute_reward(self, fused_01, ir_01, vi_01):
        """
        计算 CLIP 奖励
        Returns: r_sim (相似度), r_prompt (方向引导度) - 范围均为 0~1
        """
        if not self.enabled:
            return torch.zeros(fused_01.shape[0], device=self.device), torch.zeros(fused_01.shape[0],
                                                                                   device=self.device)

        with torch.no_grad():
            # 编码图像特征
            f_feat = self.model.encode_image(self._preprocess(fused_01)).float()
            i_feat = self.model.encode_image(self._preprocess(ir_01)).float()
            v_feat = self.model.encode_image(self._preprocess(vi_01)).float()

            # 归一化
            f_norm = F.normalize(f_feat, dim=-1)
            i_norm = F.normalize(i_feat, dim=-1)
            v_norm = F.normalize(v_feat, dim=-1)

            # --- A. 相似度奖励 (Similarity Reward) ---
            # 融合图与源图的特征距离
            sim_vi = torch.cosine_similarity(f_norm, v_norm, dim=-1)
            sim_ir = torch.cosine_similarity(f_norm, i_norm, dim=-1)

            # 将余弦相似度 [-1, 1] 映射到 [0, 1]
            r_sim = 0.5 * ((sim_vi + 1) * 0.5) + 0.5 * ((sim_ir + 1) * 0.5)

            # --- B. 方向引导奖励 (Directional Prompt Reward) ---
            # 图像特征的变化方向 (Fused - Source) 是否与文本语义的变化方向 (Target - Source) 一致？

            # 计算图像变化向量
            delta_img_vi = F.normalize(f_norm - v_norm, dim=-1)
            delta_img_ir = F.normalize(f_norm - i_norm, dim=-1)

            # 计算方向一致性
            dir_score_vi = torch.cosine_similarity(delta_img_vi, self.delta_txt_vi, dim=-1)
            dir_score_ir = torch.cosine_similarity(delta_img_ir, self.delta_txt_ir, dim=-1)

            # 映射并取平均
            r_prompt_raw = 0.5 * dir_score_vi + 0.5 * dir_score_ir
            r_prompt = (r_prompt_raw + 1) * 0.5

            return r_sim, r_prompt

# =================================================================
# 👁️ VIF 独立模块 (新解耦模块)
# =================================================================
class VIFReward(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        # 初始化高斯核
        self.vif_kernels = [
            self._get_gaussian_kernel(2 ** (5 - scale) + 1, (2 ** (5 - scale) + 1) / 5).to(device)
            for scale in range(1, 5)
        ]

    def _get_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size, device=self.device).float() - (size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    def _conv2d_valid(self, img, kernel):
        pad = kernel.shape[-1] // 2
        return F.conv2d(img, kernel, padding=pad)

    def compute_local_information(self, img, kernel_size=5):
        pad = kernel_size // 2
        mean = F.avg_pool2d(img, kernel_size, stride=1, padding=pad)
        sq_mean = F.avg_pool2d(img ** 2, kernel_size, stride=1, padding=pad)
        var = torch.relu(sq_mean - mean ** 2)
        return var

    def to_gray(self, x):
        if x.shape[1] == 3:
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return x

    def _vif_single(self, ref, dist, return_map=False):
        sigma_nsq = 2.0
        eps = 1e-10
        vif_map_total = torch.zeros_like(ref)
        curr_ref, curr_dist = ref, dist
        for i, kernel in enumerate(self.vif_kernels):
            if i > 0:
                curr_ref = self._conv2d_valid(curr_ref, kernel)[:, :, ::2, ::2]
                curr_dist = self._conv2d_valid(curr_dist, kernel)[:, :, ::2, ::2]
            mu1 = self._conv2d_valid(curr_ref, kernel)
            mu2 = self._conv2d_valid(curr_dist, kernel)
            mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
            sigma1_sq = F.relu(self._conv2d_valid(curr_ref ** 2, kernel) - mu1_sq)
            sigma2_sq = F.relu(self._conv2d_valid(curr_dist ** 2, kernel) - mu2_sq)
            sigma12 = self._conv2d_valid(curr_ref * curr_dist, kernel) - mu1_mu2
            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps
            sigma1_sq[sigma1_sq < eps] = eps
            g_term = torch.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq))
            dist_term = torch.log10(1 + sigma1_sq / sigma_nsq)
            if return_map:
                scale_map = g_term / (dist_term + eps)
                scale_map = torch.clamp(scale_map, max=10.0)
                if i > 0:
                    scale_map = F.interpolate(scale_map, size=ref.shape[2:], mode='bilinear', align_corners=False)
                vif_map_total += scale_map
        if return_map:
            return vif_map_total
        else:
            return 0.0

    def compute_reward(self, fused_01, ir_01, vi_01):
        """
        计算 VIF 奖励
        Input: Normalized tensors [0, 1]
        Output: Raw VIF score (scalar per batch, but here summed over batch/pixels, handled by FusionReward for normalization)
        """
        # VIFF 算法基于 [0, 255] 的像素值假设 (因为 sigma_nsq=2.0)
        # 所以在内部进行灰度化和反归一化
        fused_gray = self.to_gray(fused_01) * 255.0
        ir_gray = self.to_gray(ir_01) * 255.0
        vi_gray = self.to_gray(vi_01) * 255.0

        map_ir = self._vif_single(ir_gray, fused_gray, return_map=True)
        map_vi = self._vif_single(vi_gray, fused_gray, return_map=True)

        info_ir = self.compute_local_information(ir_gray)
        info_vi = self.compute_local_information(vi_gray)
        sum_info = info_ir + info_vi + 1e-6

        w_ir = info_ir / sum_info
        w_vi = info_vi / sum_info

        # 过滤背景噪声
        thr = 0.05 * sum_info.mean(dim=[1, 2, 3], keepdim=True)
        valid_mask = (sum_info > thr).float()

        weighted_vif = w_ir * map_ir + w_vi * map_vi
        numerator = (weighted_vif * valid_mask).sum(dim=[1, 2, 3])
        valid_pixels = valid_mask.sum(dim=[1, 2, 3])
        denominator = torch.clamp(valid_pixels, min=1.0)

        final_score = numerator / denominator
        return final_score

# =================================================================
# 🌈 Main Fusion Reward (现在非常干净)
# =================================================================
# class FusionReward(nn.Module):
#     def __init__(self, device="cuda", use_clip_reward=True, dinov3_strategy="select_3_6_9_11"):
#         super().__init__()
#         self.device = device
#
#         # 1. CLIP Module
#         self.clip = CLIPLogicReward(device=device) if use_clip_reward else None
#         # 2. DINOv3 Module
#         self.dinov3 = DINOv3LocalReward(device=device, strategy=dinov3_strategy)
#
#         # 3. VIF Module (现在是独立的)
#         self.vif = VIFReward(device=device)
#
#     def compute_color_reward(self, fused, vi):
#         f_down = F.interpolate(fused, size=(128, 128), mode='area')
#         v_down = F.interpolate(vi, size=(128, 128), mode='area')
#         _, u_f, v_f = self.rgb_to_yuv(f_down)
#         _, u_v, v_v = self.rgb_to_yuv(v_down)
#         color_diff = torch.abs(u_f - u_v).mean(dim=[1, 2]) + torch.abs(v_f - v_v).mean(dim=[1, 2])
#         r_color = torch.exp(-5.0 * color_diff)
#         return r_color
#
#     def rgb_to_yuv(self, image):
#         r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
#         y = 0.299 * r + 0.587 * g + 0.114 * b
#         u = -0.147 * r - 0.289 * g + 0.436 * b
#         v = 0.615 * r - 0.515 * g - 0.100 * b
#         return y, u, v
#
#     def compute_reward(self, fused, ir, vi):
#         # 1. 预处理 (Normalization -> 0~1)
#         if fused.min() < 0:
#             fused_01 = torch.clamp((fused + 1) * 0.5, 0, 1)
#             ir_01 = torch.clamp((ir + 1) * 0.5, 0, 1)
#             vi_01 = torch.clamp((vi + 1) * 0.5, 0, 1)
#         else:
#             fused_01, ir_01, vi_01 = torch.clamp(fused, 0, 1), torch.clamp(ir, 0, 1), torch.clamp(vi, 0, 1)
#
#         fused_rgb = fused_01.repeat(1, 3, 1, 1) if fused_01.shape[1] == 1 else fused_01
#         vi_rgb = vi_01.repeat(1, 3, 1, 1) if vi_01.shape[1] == 1 else vi_01
#         ir_rgb = ir_01.repeat(1, 3, 1, 1) if ir_01.shape[1] == 1 else ir_01
#
#         # -----------------------------------------------
#         # [A] VIFF (调用独立模块)
#         # -----------------------------------------------
#         r_vif = self.vif.compute_reward(fused_01, ir_01, vi_01)
#         r_vif_norm = torch.tanh(r_vif / 2.5)  # 保持原有的归一化逻辑
#         # -----------------------------------------------
#         # [B] DINOv3 (语义一致性奖励 - MAX Strategy) 🦕
#         # -----------------------------------------------
#         if self.dinov3.has_model:
#             # 提取所有特征
#             feats_f = self.dinov3.get_all_layer_features(fused_rgb)
#             feats_i = self.dinov3.get_all_layer_features(ir_rgb)
#             feats_v = self.dinov3.get_all_layer_features(vi_rgb)
#
#             # 计算奖励 (内部现在使用 Max 融合)
#             r_sem = self.dinov3.compute_semantic_from_features(feats_f, feats_i, feats_v)
#         else:
#             r_sem = torch.zeros_like(r_vif_norm)
#         # -----------------------------------------------
#         # [C] CLIP
#         # -----------------------------------------------
#         if self.clip and self.clip.enabled:
#             r_sim, r_prompt = self.clip.compute_reward(fused_rgb, ir_rgb, vi_rgb)
#             r_clip = 0.3 * r_sim + 0.7 * r_prompt
#         else:
#             r_clip = torch.zeros_like(r_vif_norm)
#         r_uv = self.compute_color_reward(fused_rgb, vi_rgb)
#         # -----------------------------------------------
#         # 4. 最终加权 (Weighting)
#         # -----------------------------------------------
#         w_vif = 1.0
#         w_clip = 1.0
#         w_sem = 1.0
#         w_uv = 0.7
#         total_reward = w_vif * r_vif_norm + \
#                        w_sem * r_sem + \
#                        w_clip * r_clip + \
#                        w_uv * r_uv
#
#         # 1. Info (Scalars)
#         reward_info = {
#             "Total": total_reward.mean().item(),
#             "VIF": r_vif.mean().item(),
#             "CLIP": r_clip.mean().item(),
#             "DINO": r_sem.mean().item()
#         }
#
#         # 2. Sub Rewards (Tensors)
#         sub_rewards = {
#             "VIF": r_vif_norm.detach(),
#             "CLIP": r_clip.detach(),
#             "DINO": r_sem.detach()
#         }
#
#         return total_reward, reward_info, sub_rewards

# =================================================================
# 🌈 Main Fusion Reward (Updated with YCbCr Strategy)
# =================================================================
class FusionReward(nn.Module):
    def __init__(self, device="cuda", use_clip_reward=True, dinov3_strategy="select_3_6_9_11"):
        super().__init__()
        self.device = device

        # 1. CLIP Module
        self.clip = CLIPLogicReward(device=device) if use_clip_reward else None
        # 2. DINOv3 Module
        self.dinov3 = DINOv3LocalReward(device=device, strategy=dinov3_strategy)
        # 3. VIF Module
        self.vif = VIFReward(device=device)

    def rgb_to_cbcr(self, image):
        """
        将 RGB 转换为 YCbCr，并仅返回 Cb 和 Cr 通道。
        基于 ITU-R BT.601 标准。
        Input: [B, 3, H, W], range [0, 1]
        """
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        # Y  =  0.299 * R + 0.587 * G + 0.114 * B (这里不需要 Y)
        # Cb = -0.1687 * R - 0.3313 * G + 0.500 * B
        # Cr =  0.500 * R - 0.4187 * G - 0.0813 * B

        cb = -0.168736 * r - 0.331264 * g + 0.50000 * b
        cr = 0.500000 * r - 0.418688 * g - 0.081312 * b

        return cb, cr

    def compute_color_reward(self, fused, vi):
        """
        计算色度一致性奖励 (只关注 Cb, Cr)
        """
        # 下采样以关注全局色彩分布，忽略局部高频噪声
        f_down = F.interpolate(fused, size=(128, 128), mode='area')
        v_down = F.interpolate(vi, size=(128, 128), mode='area')

        # 提取 Cb, Cr
        f_cb, f_cr = self.rgb_to_cbcr(f_down)
        v_cb, v_cr = self.rgb_to_cbcr(v_down)

        # 计算 L1 距离
        # 融合图的色度 应该 接近 可见光图的色度
        diff_cb = torch.abs(f_cb - v_cb).mean(dim=[1, 2])
        diff_cr = torch.abs(f_cr - v_cr).mean(dim=[1, 2])

        total_chroma_diff = diff_cb + diff_cr

        # 转换为奖励: 差异越小，奖励越大 (0~1)
        # 系数 10.0 可以根据实际训练情况调整，差异通常很小，所以系数大一点
        r_color = torch.exp(-10.0 * total_chroma_diff)
        return r_color

    def compute_reward(self, fused, ir, vi):
        # 1. 预处理 (Normalization -> 0~1)
        if fused.min() < 0:
            fused_01 = torch.clamp((fused + 1) * 0.5, 0, 1)
            ir_01 = torch.clamp((ir + 1) * 0.5, 0, 1)
            vi_01 = torch.clamp((vi + 1) * 0.5, 0, 1)
        else:
            fused_01, ir_01, vi_01 = torch.clamp(fused, 0, 1), torch.clamp(ir, 0, 1), torch.clamp(vi, 0, 1)

        fused_rgb = fused_01.repeat(1, 3, 1, 1) if fused_01.shape[1] == 1 else fused_01
        vi_rgb = vi_01.repeat(1, 3, 1, 1) if vi_01.shape[1] == 1 else vi_01
        ir_rgb = ir_01.repeat(1, 3, 1, 1) if ir_01.shape[1] == 1 else ir_01

        # -----------------------------------------------
        # [A] VIFF
        # -----------------------------------------------
        r_vif = self.vif.compute_reward(fused_01, ir_01, vi_01)
        r_vif_norm = torch.tanh(r_vif / 2.5)

        # -----------------------------------------------
        # [B] DINOv3
        # -----------------------------------------------
        if self.dinov3.has_model:
            feats_f = self.dinov3.get_all_layer_features(fused_rgb)
            feats_i = self.dinov3.get_all_layer_features(ir_rgb)
            feats_v = self.dinov3.get_all_layer_features(vi_rgb)
            r_sem = self.dinov3.compute_semantic_from_features(feats_f, feats_i, feats_v)
        else:
            r_sem = torch.zeros_like(r_vif_norm)

        # -----------------------------------------------
        # [C] CLIP
        # -----------------------------------------------
        if self.clip and self.clip.enabled:
            r_sim, r_prompt = self.clip.compute_reward(fused_rgb, ir_rgb, vi_rgb)
            r_clip = 0.3 * r_sim + 0.7 * r_prompt
        else:
            r_clip = torch.zeros_like(r_vif_norm)

        # -----------------------------------------------
        # [D] CbCr Color Reward (New!)
        # -----------------------------------------------
        r_color = self.compute_color_reward(fused_rgb, vi_rgb)

        # -----------------------------------------------
        # 4. 最终加权 (Weighting)
        # -----------------------------------------------
        w_vif = 1.0
        w_sem = 1.0
        w_clip = 1.0
        w_color = 0.5  # 可以稍微调高一点，因为现在不惩罚亮度差异了

        total_reward = w_vif * r_vif_norm + \
                       w_sem * r_sem + \
                       w_clip * r_clip + \
                       w_color * r_color

        # 1. Info (Scalars)
        reward_info = {
            "Total": total_reward.mean().item(),
            "VIF": r_vif.mean().item(),
            "CLIP": r_clip.mean().item(),
            "DINO": r_sem.mean().item(),
            "Color": r_color.mean().item()
        }

        # 2. Sub Rewards (Tensors)
        sub_rewards = {
            "VIF": r_vif_norm.detach(),
            "CLIP": r_clip.detach(),
            "DINO": r_sem.detach(),
            "Color": r_color.detach()
        }

        return total_reward, reward_info, sub_rewards
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionReward(device)

    print("Testing RGB Input...")
    dummy_rgb = torch.randn(2, 3, 256, 256).to(device)
    dummy_ir = torch.randn(2, 3, 256, 256).to(device)
    dummy_vi = torch.randn(2, 3, 256, 256).to(device)

    # 模拟调用
    r, info, subs = model.compute_reward(dummy_rgb, dummy_ir, dummy_vi)
    print("Log Info:", info)
    print("Sub Tensor Keys:", subs.keys())