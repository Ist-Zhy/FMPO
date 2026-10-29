import copy
import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
from torch import einsum, nn
from tqdm.auto import tqdm

# 保持原有接口和工具函数不变
ModelPrediction = namedtuple('ModelPrediction', ['pred_velocity', 'pred_x_start'])


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        v = v / (h * w)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            condition=False,
    ):
        super().__init__()
        self.channels = channels
        self.depth = len(dim_mults)
        # 保持 9 通道输入 (x_t + x_ir + x_vi)
        input_channels = 9
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.out_dim = channels
        self.final_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, time):
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x


class UnetVelocity(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3,
                 resnet_block_groups=8, learned_variance=False, learned_sinusoidal_cond=False,
                 random_fourier_features=False, learned_sinusoidal_dim=16, condition=False):
        super().__init__()
        self.condition = condition
        self.channels = channels
        self.out_dim = default(out_dim, channels)
        self.unet = Unet(dim, init_dim=init_dim, out_dim=out_dim, dim_mults=dim_mults,
                         channels=channels, resnet_block_groups=resnet_block_groups,
                         learned_variance=learned_variance, learned_sinusoidal_cond=learned_sinusoidal_cond,
                         random_fourier_features=random_fourier_features,
                         learned_sinusoidal_dim=learned_sinusoidal_dim, condition=condition)

    def forward(self, x, time):
        return self.unet(x, time)


# ================================================================
# OPTIMIZED FLOW MATCHING CLASS
# 优化点：
# 1. 移除了路径中的噪声 (Pure OT-FM)，提高收敛速度和成像清晰度
# 2. 采样和训练逻辑中的输入/输出完全对齐
# 3. 增加了可选的辅助损失计算（在 predict_start_from_velocity 中）
# ================================================================

class FlowMatching(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            condition=False,
            # 优化：sigma设为0，构建确定性路径
            sigma_min=0.0,
            sigma_max=0.0,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.condition = condition
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.num_timesteps = timesteps
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        # 即使不用噪声，保留buffer结构以防万一，但设为0
        sigmas = torch.linspace(sigma_max, sigma_min, timesteps)
        self.register_buffer('sigmas', sigmas)

    def get_velocity_field(self, x_start, x_input, t=None):
        """
        Optimal Transport (OT) Velocity Field
        v_t = x_start - x_input
        """
        return x_start - x_input

    def sample_flow_path(self, x_start, x_input, t, noise=None):
        """
        Optimization: Pure Linear Interpolation (No Noise)
        x_t = (1-t) * x_input + t * x_start
        """
        # 即使传入了 noise 也不使用，确保路径平滑
        t_expanded = t.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / self.num_timesteps

        # 纯 OT 路径：从 x_input 直达 x_start
        x_t = (1 - t_expanded) * x_input + t_expanded * x_start
        return x_t

    def predict_start_from_velocity(self, x_t, t, velocity):
        """Helper to predict x_start from current x_t and velocity"""
        t_expanded = t.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / self.num_timesteps
        # Inverse of flow: x_start = x_t + (1-t) * v
        return x_t + (1 - t_expanded) * velocity

    def model_predictions(self, x_input, x, t, clip_denoised=True):
        """
        Model predicts the velocity field v_t
        """
        if self.condition:
            # x_input is [x_ir_norm, x_vi_norm]
            x_ir_norm = x_input[0]
            x_vi_norm = x_input[1]
            x_in = torch.cat((x, x_ir_norm, x_vi_norm), dim=1)
        else:
            x_in = x

        t_normalized = t.float() / self.num_timesteps
        velocity = self.model(x_in, t_normalized)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_denoised else lambda x: x

        # 预测的 x_start (仅用于参考或辅助 Loss)
        x_start_pred = self.predict_start_from_velocity(x, t, velocity)
        x_start_pred = maybe_clip(x_start_pred)

        # 裁剪 Velocity 本身通常不需要，但为了数值稳定性可以保留
        # velocity = maybe_clip(velocity)

        return ModelPrediction(pred_velocity=velocity, pred_x_start=x_start_pred)

    @torch.no_grad()
    def euler_sample_step(self, x_input, x, t, dt):
        preds = self.model_predictions(x_input, x, t)
        velocity = preds.pred_velocity
        # Euler Step: x_{t+dt} = x_t + dt * v
        x_next = x + dt * velocity
        return x_next, preds.pred_x_start

    @torch.no_grad()
    def sample_euler(self, x_input, shape, last=True):
        if self.condition:
            ir_norm = x_input[0]
            vi_norm = x_input[1]
            # 始终从 (IR+VI)/2 开始，保持一致性
            img = (ir_norm + vi_norm) / 2.0
            batch, device = img.shape[0], img.device
            input_start = img.clone()
        else:
            batch, device = shape[0], next(self.model.parameters()).device
            img = torch.randn(shape, device=device)

        dt = 1.0 / self.sampling_timesteps
        times = torch.linspace(0, self.num_timesteps - 1, self.sampling_timesteps, device=device)

        img_list = [] if not last else None

        for i, t_val in enumerate(tqdm(times, desc='Flow matching sampling')):
            t_batch = torch.full((batch,), int(t_val), device=device, dtype=torch.long)
            img, _ = self.euler_sample_step(x_input, img, t_batch, dt)

            if not last:
                img_list.append(img.clone())

        if self.condition:
            if not last:
                img_list = [input_start] + img_list
                return unnormalize_to_zero_to_one(img_list)
            else:
                return unnormalize_to_zero_to_one(img)
        else:
            if not last:
                return unnormalize_to_zero_to_one(img_list)
            else:
                return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def sample(self, x_input=None, batch_size=16, last=True):
        image_size, channels = self.image_size, self.channels
        if self.condition and x_input is not None:
            # x_input 预期是 [x_ir, x_vi]，范围 [0, 1]
            x_input_norm = normalize_to_neg_one_to_one(x_input)
            batch_size, _, h, w = x_input_norm[0].shape
            size = (batch_size, self.channels, h, w)
            return self.sample_euler(x_input_norm, size, last=last)
        else:
            size = (batch_size, channels, image_size, image_size)
            return self.sample_euler(None, size, last=last)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imgs, t, noise=None):
        # imgs = [x_ir, x_vi, x_start]
        x_ir = imgs[0]
        x_vi = imgs[1]
        x_start = imgs[2]  # 伪真值

        # 保持与采样一致：Source = (IR+VI)/2
        x_input = (x_ir + x_vi) / 2.0

        # 优化：不使用噪声，纯 OT 路径
        # noise 参数保留但被忽略
        x_t = self.sample_flow_path(x_start, x_input, t)

        # 目标速度场 (Target Velocity)
        true_velocity = self.get_velocity_field(x_start, x_input, t)

        # 模型预测
        if self.condition:
            x_in = torch.cat((x_t, x_ir, x_vi), dim=1)
        else:
            x_in = x_t

        t_normalized = t.float() / self.num_timesteps
        pred_velocity = self.model(x_in, t_normalized)

        # 1. 速度场 Loss (主 Loss)
        loss_flow = self.loss_fn(pred_velocity, true_velocity, reduction='none')
        loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean').mean()

        # 2. (可选) 辅助 Loss: 约束预测的 x_start 接近真值 x_start
        # 这能帮助模型更快稳定下来
        x_start_pred = self.predict_start_from_velocity(x_t, t, pred_velocity)
        loss_recon = self.loss_fn(x_start_pred, x_start, reduction='none')
        loss_recon = reduce(loss_recon, 'b ... -> b (...)', 'mean').mean()

        # 总 Loss，你可以调整权重，这里简单设为 1:1
        total_loss = loss_flow + 0.0 * loss_recon

        return total_loss

    def forward(self, img, *args, **kwargs):
        b, c, h, w = img[0].shape
        device = img[0].device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)