from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
# from utils.utils_color import RGB_HSV, RGB_YCbCr
from utils.loss_ssim import ssim
import torchvision.transforms.functional as TF


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, layers=['relu3_3']):
        super(VGGPerceptualLoss, self).__init__()
        self.device = device
        vgg = models.vgg19(pretrained=False).to(self.device)  # 不加载默认的预训练模型
        vgg.load_state_dict(torch.load("./checkpoint/vgg19-dcbb9e9d.pth"))  # 加载本地下载的权重
        # 提取VGG19的特征层
        vgg_pretrained = vgg.features[:36].eval()

        for param in vgg_pretrained.parameters():
            param.requires_grad = False
        self.vgg = vgg_pretrained
        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 17,
            'relu4_3': 26,
        }
        self.selected_layers = [self.layer_map[l] for l in layers]
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.selected_layers:
                loss += F.l1_loss(x, y)
        return loss


class Fusion_loss_(nn.Module):
    def __init__(self):
        super(Fusion_loss_, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

    def forward(self, image_A, image_B, image_fused):
        # 提取 Y 通道（灰度）
        image_fused = unnormalize_to_zero_to_one(image_fused)
        image_A = unnormalize_to_zero_to_one(image_A)
        image_B = unnormalize_to_zero_to_one(image_B)
        image_A_Y, _, _ = RGB2YCbCr(image_A)
        image_B_Y, _, _ = RGB2YCbCr(image_B)
        image_fused_Y, _, _ = RGB2YCbCr(image_fused)

        # IVF
        IVF_F = image_fused[:8, :,:,:]
        IVF_VI = image_B[:8,:,:,:]
        IVF_IR = image_A[:8,:,:,:]
        # MIF

        loss_l1 = 4 * self.L_Inten(image_A_Y, image_B_Y, image_fused_Y)
        loss_gradient = 5 * self.L_Grad(image_A_Y, image_B_Y, image_fused_Y)
        loss_SSIM = 1 * (1 - self.L_SSIM(image_A_Y, image_B_Y, image_fused_Y))

        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss


class Fusion_loss_L(nn.Module):
    def __init__(self, tau=0.5):
        super(Fusion_loss_L, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.tau = tau

    def forward(self, image_A, image_B, image_fused, fu, modality=None):
        # image_fused = unnormalize_to_zero_to_one(image_fused)
        # image_A = unnormalize_to_zero_to_one(image_A)
        # image_B = unnormalize_to_zero_to_one(image_B)

        A_Y, A_Cb, A_Cr = RGB2YCbCr(image_A)
        B_Y, B_Cb, B_Cr = RGB2YCbCr(image_B)
        F_Y, F_Cb, F_Cr = RGB2YCbCr(image_fused)
        FGT_Y, FGT_Cb, FGT_Cr = RGB2YCbCr(fu)
        # === 灰度损失 ===
        loss_l1 = 5 * self.L_Inten(A_Y, B_Y, F_Y)
        loss_grad = 10 * self.L_Grad(A_Y, B_Y, F_Y)
        loss_ssim = 1 * (1 - self.L_SSIM(A_Y, B_Y, F_Y))
        fusion_loss = loss_l1 + loss_grad + loss_ssim

        # === 彩色模态才加色彩一致性损失 ===
        if modality in ['MEF', 'MIF', 'MFF']:
            # 计算色彩加权融合
            Cb1, Cb2 = A_Cb, B_Cb
            Cr1, Cr2 = A_Cr, B_Cr

            Cb_fused_gt = (Cb1 * torch.abs(Cb1 - self.tau) + Cb2 * torch.abs(Cb2 - self.tau)) / \
                          (torch.abs(Cb1 - self.tau) + torch.abs(Cb2 - self.tau) + 1e-8)
            Cr_fused_gt = (Cr1 * torch.abs(Cr1 - self.tau) + Cr2 * torch.abs(Cr2 - self.tau)) / \
                          (torch.abs(Cr1 - self.tau) + torch.abs(Cr2 - self.tau) + 1e-8)

            color_loss = F.l1_loss(F_Cb, Cb_fused_gt) + F.l1_loss(F_Cr, Cr_fused_gt)
            fusion_loss += 2 * color_loss  # 权重可调
        else:
            Cb1, Cb2 = FGT_Cb, B_Cb
            Cr1, Cr2 = FGT_Cr, B_Cr
            Cb_fused_gt = (Cb1 * torch.abs(Cb1 - self.tau) + Cb2 * torch.abs(Cb2 - self.tau)) / \
                          (torch.abs(Cb1 - self.tau) + torch.abs(Cb2 - self.tau) + 1e-8)
            Cr_fused_gt = (Cr1 * torch.abs(Cr1 - self.tau) + Cr2 * torch.abs(Cr2 - self.tau)) / \
                          (torch.abs(Cr1 - self.tau) + torch.abs(Cr2 - self.tau) + 1e-8)
            color_loss = F.l1_loss(F_Cb, Cb_fused_gt) + F.l1_loss(F_Cr, Cr_fused_gt)
            fusion_loss += 2 * color_loss
        return fusion_loss


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

def RGB2YCbCr(img):
    if img.shape[1] != 3:
        raise ValueError("输入图像必须是3通道的RGB图像")
    R = img[:, 0:1, :, :]
    G = img[:, 1:2, :, :]
    B = img[:, 2:3, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5
    return Y, Cb, Cr

class sobel_operation(nn.Module):
    def __init__(self):
        super(sobel_operation, self).__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(sobel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return grad

class Fusion_mask_loss(nn.Module):
    def __init__(self):
        super(Fusion_mask_loss, self).__init__()
        print('Using Fusion_mask_loss() with IR>VIS mask for first 4 batches~')
        self.sobelconv = sobel_operation()
        self.loss_func = nn.L1Loss(reduction='mean')
        self.L_SSIM = L_SSIM()
    def forward(self, imgs_fusion, img_A, img_B):
        # 取最后一层融合输出

        # 全部 unnormalize 到 [0, 1]
        # img_fusion = unnormalize_to_zero_to_one(imgs_fusion)
        # img_A = unnormalize_to_zero_to_one(img_A)
        # img_B = unnormalize_to_zero_to_one(img_B)

        # 转换为 YCbCr
        Y_fusion, Cb_fusion, Cr_fusion = RGB2YCbCr(imgs_fusion)
        Y_A, Cb_A, Cr_A = RGB2YCbCr(img_A)
        Y_B, Cb_B, Cr_B = RGB2YCbCr(img_B)

        # 计算强度参考
        Y_joint = torch.max(Y_A, Y_B)

        # 生成 mask：只对前4个 batch 做
        mask = (Y_A[:4] > Y_B[:4]).float()

        # -------------------------
        # 强度损失
        loss_intensity_masked = (
            10 * self.loss_func(Y_fusion[:4], Y_joint[:4]) +
            40 * self.loss_func(mask * Y_fusion[:4], mask * Y_A[:4]) +
            5  * self.loss_func((1 - mask) * Y_fusion[:4], (1 - mask) * Y_B[:4])
        )
        # 后面5个 batch（不使用 mask），普通保真
        loss_intensity_unmasked = 30 * self.loss_func(Y_fusion[4:], Y_joint[4:])

        loss_intensity = loss_intensity_masked + loss_intensity_unmasked
        loss_SSIM = 1 * (1 - self.L_SSIM(Y_A[4:], Y_B[4:], Y_fusion[4:]))
        # -------------------------
        # 色彩损失（只对前4个 batch 的可见光部分）
        loss_color = (
            self.loss_func((1 - mask) * Cb_fusion[:4], (1 - mask) * Cb_B[:4]) +
            self.loss_func((1 - mask) * Cr_fusion[:4], (1 - mask) * Cr_B[:4])
        )

        # -------------------------
        # 梯度损失
        grad_A = self.sobelconv(Y_A)
        grad_B = self.sobelconv(Y_B)
        grad_fusion = self.sobelconv(Y_fusion)
        grad_joint = torch.max(grad_A, grad_B)

        # 前4个：用 mask 加权
        loss_grad_masked = (
            10 * self.loss_func(grad_fusion[:4], grad_joint[:4]) +
            50 * self.loss_func(mask * grad_fusion[:4], mask * grad_A[:4])
        )
        # 后5个：普通 gradient match
        loss_grad_unmasked = 30 * self.loss_func(grad_fusion[4:], grad_joint[4:])

        loss_grad = loss_grad_masked + loss_grad_unmasked

        # -------------------------
        # 总 loss
        loss_fusion = 4*loss_intensity + 1*loss_color + 4*loss_grad + 1 * loss_SSIM

        return {
            'loss_intensity': loss_intensity,
            'loss_color': loss_color,
            'loss_grad': loss_grad,
            'loss_fusion': loss_fusion
        }

class Fusion_loss_T(nn.Module):
    def __init__(self, device):
        super(Fusion_loss_T, self).__init__()
        self.device = device
        self.sobelconv = sobel_operation().to(device)
        self.loss_func = nn.L1Loss(reduction='mean').to(device)
        self.L_SSIM = L_SSIM().to(device)

        # 为不同模态定义权重配置
        self.modality_weights = {
            'IVF': {
                'intensity_joint': 20, 'intensity_gt': 10,
                'grad_joint': 100, 'grad_gt': 10,
                'ssim': 1
            },
            'MEF': {
                'intensity_joint': 15, 'intensity_gt': 35,
                'grad_joint': 60, 'grad_gt': 25,
                'ssim': 1
            },
            'MIF': {
                'intensity_joint': 20, 'intensity_gt': 30,
                'grad_joint': 50, 'grad_gt': 20,
                'ssim': 1
            },
            'MFF': {
                'intensity_joint': 20, 'intensity_gt': 30,
                'grad_joint': 45, 'grad_gt': 20,
                'ssim': 1.5
            }
        }

        # 默认权重
        self.default_weights = {
            'intensity_joint': 20, 'intensity_gt': 30,
            'grad_joint': 50, 'grad_gt': 20,
            'ssim': 1
        }

    def forward(self, img_A, img_B, img_fu, GT_fu, modality=None):
        """
        计算融合损失

        参数:
            img_A: 红外源图像 (ir)
            img_B: 可见光源图像 (vi)
            img_fu: 模型输出的融合图像
            GT_fu: 目标融合图像
            modality: 数据模态名称 (IVF, MEF, MIF, MFF)

        返回:
            包含各项损失的字典
        """
        # 确保输入也都在目标 device 上
        img_A = img_A.to(self.device)
        img_B = img_B.to(self.device)
        img_fu = img_fu.to(self.device)
        GT_fu = GT_fu.to(self.device)

        # 根据模态选择权重
        if modality and modality in self.modality_weights:
            weights = self.modality_weights[modality]
        else:
            weights = self.default_weights

        # 强度损失
        Y_joint = torch.max(img_A, img_B)
        loss_intensity = (
                weights['intensity_joint'] * self.loss_func(img_fu, Y_joint) +
                weights['intensity_gt'] * self.loss_func(img_fu, GT_fu)
        )

        # 梯度损失
        grad_A = self.sobelconv(img_A)
        grad_B = self.sobelconv(img_B)
        grad_fu = self.sobelconv(img_fu)
        grad_gt = self.sobelconv(GT_fu)
        grad_joint = torch.max(grad_A, grad_B)
        loss_grad = (
                weights['grad_joint'] * self.loss_func(grad_fu, grad_joint) +
                weights['grad_gt'] * self.loss_func(grad_fu, grad_gt)
        )

        # SSIM损失
        loss_SSIM = weights['ssim'] * (1 - self.L_SSIM(img_A, img_B, img_fu))

        # 总体损失
        loss_fusion = loss_intensity + loss_grad + loss_SSIM

        return {
            'loss_intensity': loss_intensity,
            'loss_grad': loss_grad,
            'loss_ssim': loss_SSIM,
            'loss_fusion': loss_fusion
        }
