import copy
import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import torchvision.transforms as transforms
import Augmentor
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
import time
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm
import csv
from Dataset.Restoration_MMIF import Restoration
from utils import metrics
from utils.Evaluator import *
import logging

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def tensor_to_gray(img_tensor):
    """
    将输出的 torch.Tensor 图像 (C, H, W)，转为 OpenCV 灰度图（H, W）
    img_tensor: tensor, shape: [C, H, W], range: [0, 1] or [0, 255]
    return: gray image as numpy array, shape: [H, W], dtype: uint8
    """
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()  # [H, W, C]
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = np.round(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    return gray


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def cycle(dl):
    while True:
        for data in dl:
            yield data


def exists(x):
    return x is not None


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset,
            *,
            gradient_accumulate_every=1,
            augment_flip=True,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_unet=1,
            num_samples=25,
            results_folder='./results/sample1',
            Experiment_folder='./Experiment/sample',
            amp=False,
            fp16=False,
            split_batches=True,
            convert_image_to=None,
            condition=False,
            sub_dir=False,
    ):
        super().__init__()
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.accelerator.native_amp = amp
        self.num_unet = num_unet
        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.Experiment = Experiment_folder
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition
        self.opts = dataset
        # ==========================================
        # [新增] 初始化 Loss 记录文件
        # # ==========================================
        if self.accelerator.is_main_process:
            self.loss_log_file = os.path.join(self.Experiment, 'loss_log.csv')
            # 如果文件不存在，创建并写入表头；如果存在则追加（防止断点续训覆盖）
            if not os.path.exists(self.loss_log_file):
                with open(self.loss_log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'loss'])
            logger.info(f"Loss logging to: {self.loss_log_file}")

        # 简化：只加载 IVF 任务
        if self.condition:
            self.T_IVF = None
            logger.info("Loading IVF dataset...")
            data = Restoration(config=self.opts['IVF'])
            train_loader, val_dataset = data.get_loaders()
            self.T_IVF = cycle(self.accelerator.prepare(train_loader))

            # 验证数据加载器
            self.sample_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            logger.info(f"IVF dataset loaded. Validation samples: {len(val_dataset)}")

        # 优化器
        self.opt0 = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)
            self.set_results_folder(results_folder)

        self.step = 0
        self.model, self.opt0 = self.accelerator.prepare(self.model, self.opt0)
        device = self.accelerator.device
        self.device = device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt0': self.opt0.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        logger.info(f"Model saved at milestone {milestone}")

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')
        print(f"Loading model from: {path}")
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            self.model = self.accelerator.unwrap_model(self.model)
            self.model.load_state_dict(data['model'])
            self.step = data['step']
            self.opt0.load_state_dict(data['opt0'])
            self.opt0.param_groups[0]['capturable'] = True
            self.ema.load_state_dict(data['ema'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Model file not found: {path}")
        self.ema.to(self.device)

    def get_custom_batch(self):
        """简化：只从 IVF 任务获取数据"""
        data = next(self.T_IVF)

        def squeeze_or_flatten(t):
            return t.flatten(start_dim=0, end_dim=1) if t.ndim == 5 else t

        ir = squeeze_or_flatten(data['ir']).to(self.device)
        vi = squeeze_or_flatten(data['vi']).to(self.device)
        fu = squeeze_or_flatten(data['fu']).to(self.device)

        return [ir, vi, fu]

    def train(self):
        arrows = "=" * 35
        print(f"{arrows} => 🚀 TRAINING START <= 🔥 {arrows[::-1]}")
        accelerator = self.accelerator

        with tqdm(initial=self.step, total=self.train_num_steps,
                  disable=not accelerator.is_main_process, ncols=100) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        data = self.get_custom_batch()

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        # FlowMatching 返回单个 loss，不是列表
                        if isinstance(loss, list):
                            loss = loss[0]
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                # 梯度裁剪
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                accelerator.wait_for_everyone()

                # 优化器步骤
                self.opt0.step()
                self.opt0.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process and self.step % 10 == 0:
                    # 使用 'a' (append) 模式追加写入
                    with open(self.loss_log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.step, total_loss])
                # EMA 更新和保存
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    # 定期验证和保存
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.validation(milestone)

                        # 每10个milestone保存一次模型
                        if self.step % (self.save_and_sample_every * 10) == 0:
                            self.save(milestone)


                pbar.set_description(f'loss: {total_loss:.4f}')
                pbar.update(1)

        accelerator.print('Training complete!')

    def validation(self, milestone, last=True):
        """验证函数：计算指标并保存结果"""
        self.ema.ema_model.eval()

        # 创建输出目录
        output_dir = os.path.join(self.Experiment, "outputs", f"milestone_{milestone}")
        os.makedirs(output_dir, exist_ok=True)

        avg_EN = 0.0
        avg_SD = 0.0
        avg_AG = 0.0
        total_time = 0.0
        total_imgs = 0

        now = milestone * self.save_and_sample_every

        with torch.no_grad():
            logger.info(f"\nStarting validation at step: {now}")

            for idx, (x, y) in enumerate(self.sample_loader):
                # 展平批次维度
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                _, _, h, w = x.shape

                # Padding to multiple of 16
                multiple = 16
                crop_height = int(multiple * np.ceil(h / multiple))
                crop_width = int(multiple * np.ceil(w / multiple))
                x = F.pad(x, (0, crop_width - w, 0, crop_height - h), mode='reflect')

                # 分离 IR 和 VI
                x_ir = x[:, :3, :, :].to(self.device)
                x_vi = x[:, 3:, :, :].to(self.device)

                # 采样
                start_time = time.time()
                images_list = self.ema.ema_model.sample(
                    [x_ir, x_vi], batch_size=n, last=last
                )
                elapsed_time = time.time() - start_time
                if isinstance(images_list, list):
                    final_res = images_list[-1]
                else:
                    final_res = images_list
                if final_res.dim() == 3:
                    final_res = final_res.unsqueeze(0)
                # 获取预测结果并裁剪回原始大小
                pred_x = final_res[:, :, :h, :w]
                for i in range(n):
                    # 保存图像
                    img_name = f"{y[i][:-4]}_fake.png"
                    save_path = os.path.join(output_dir, img_name)
                    utils.save_image(pred_x[i], save_path)

                    # 转换为灰度图计算指标
                    img = tensor_to_gray(pred_x[i])

                    # 计算指标
                    en_val = Evaluator.EN(img)
                    sd_val = Evaluator.SD(img)
                    ag_val = Evaluator.AG(img)

                    # 累加
                    avg_EN += en_val
                    avg_SD += sd_val
                    avg_AG += ag_val
                    total_imgs += 1
                    total_time += elapsed_time / n

                    # 打印每张图像的指标
                    print(f"[{y[i]}] EN: {en_val:.4f}, SD: {sd_val:.4f}, AG: {ag_val:.4f}")

        # 计算平均值
        if total_imgs > 0:
            avg_EN /= total_imgs
            avg_SD /= total_imgs
            avg_AG /= total_imgs
            avg_time = total_time / total_imgs

        # 打印汇总结果
        print("\n" + "╭" + "━" * 45 + "╮")
        print(f"   VALIDATION RESULTS (Milestone {milestone}, Step {now})")
        print(f"   Total Images: {total_imgs}")
        print(f"   Average EN: {avg_EN:.4f}")
        print(f"   Average SD: {avg_SD:.4f}")
        print(f"   Average AG: {avg_AG:.4f}")
        print(f"   Average Time: {avg_time:.4f}s per image")
        print("╰" + "━" * 45 + "╯\n")

        arrows = "=" * 35
        print(f"{arrows} => 🚀 PROCESSING <= 🔥 {arrows[::-1]}")

        # 切换回训练模式
        self.ema.ema_model.train()

    def test(self, milestone, last=True):
        """
        测试函数：与验证类似，但可能用于最终测试
        """
        self.ema.ema_model.eval()

        # 创建输出目录
        output_dir = os.path.join(self.Experiment, "test_outputs", f"milestone_{milestone}")
        os.makedirs(output_dir, exist_ok=True)

        avg_EN = 0.0
        avg_SD = 0.0
        avg_AG = 0.0
        total_time = 0.0
        total_imgs = 0

        now = milestone * self.save_and_sample_every

        with torch.no_grad():
            logger.info(f"\nStarting testing at step: {now}")

            for idx, (x, y) in enumerate(self.sample_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                _, _, h, w = x.shape

                # Padding
                multiple = 16
                crop_height = int(multiple * np.ceil(h / multiple))
                crop_width = int(multiple * np.ceil(w / multiple))
                x = F.pad(x, (0, crop_width - w, 0, crop_height - h), mode='reflect')

                x_ir = x[:, :3, :, :].to(self.device)
                x_vi = x[:, 3:, :, :].to(self.device)

                # 采样并计时
                start_time = time.time()
                images_list = self.ema.ema_model.sample(
                    [x_ir, x_vi], batch_size=n, last=last
                )
                elapsed_time = time.time() - start_time
                if isinstance(images_list, list):
                    final_res = images_list[-1]
                else:
                    final_res = images_list
                if final_res.dim() == 3:
                    final_res = final_res.unsqueeze(0)
                # 获取预测结果并裁剪回原始大小
                pred_x = final_res[:, :, :h, :w]
                for i in range(n):
                    # 保存
                    img_name = f"{y[i][:-4]}_fake.png"
                    save_path = os.path.join(output_dir, img_name)
                    utils.save_image(pred_x[i], save_path)

                    # 指标
                    img = tensor_to_gray(pred_x[i])
                    en_val = Evaluator.EN(img)
                    sd_val = Evaluator.SD(img)
                    ag_val = Evaluator.AG(img)

                    per_img_time = elapsed_time / n

                    print(f"[{y[i]}] EN: {en_val:.4f}, SD: {sd_val:.4f}, "
                          f"AG: {ag_val:.4f}, Time: {per_img_time:.4f}s")

                    avg_EN += en_val
                    avg_SD += sd_val
                    avg_AG += ag_val
                    total_time += per_img_time
                    total_imgs += 1

        # 汇总
        if total_imgs > 0:
            avg_EN /= total_imgs
            avg_SD /= total_imgs
            avg_AG /= total_imgs
            avg_time = total_time / total_imgs

        print("\n" + "╭" + "━" * 45 + "╮")
        print(f"   TEST RESULTS (Milestone {milestone}, Step {now})")
        print(f"   Total Images: {total_imgs}")
        print(f"   Average EN: {avg_EN:.4f}")
        print(f"   Average SD: {avg_SD:.4f}")
        print(f"   Average AG: {avg_AG:.4f}")
        print(f"   Average Time: {avg_time:.4f}s per image")
        print(f"   Total Time: {total_time:.2f}s")
        print("╰" + "━" * 45 + "╯\n")

        arrows = "=" * 35
        print(f"{arrows} => 🚀 TESTING COMPLETE <= 🔥 {arrows[::-1]}")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)
        logger.info(f"Results folder set to: {self.results_folder}")