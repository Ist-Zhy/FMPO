import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
from natsort import natsorted

class RestorationDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True, phase='train'):
        super().__init__()
        print('source dir: ', dir)
        self.phase = phase
        source_dir = dir
        if self.phase == 'train':
            dir_fu = os.path.join(source_dir, 'Fusion')
            dir_vi = os.path.join(source_dir, 'B_N')
            dir_ir = os.path.join(source_dir, 'A_N')
        else:
            dir_vi = os.path.join(source_dir, 'B_N')
            dir_ir = os.path.join(source_dir, 'A_N')
            dir_fu = os.path.join(source_dir, 'Fusion')
            if not os.path.exists(dir_fu):
                dir_fu = dir_ir
        print("VI floder: {}, IR folder: {}, Fusion folder: {}".format(dir_vi, dir_ir, dir_fu))
        IR_names, VI_names, Fu_names = [], [], []
        file_list = natsorted(os.listdir(dir_fu))
        for item in file_list:
            if item.endswith('.jpg') or item.endswith('.png') or item.endswith('.bmp'):
                IR_names.append(os.path.join(dir_ir, item))
                VI_names.append(os.path.join(dir_vi, item))
                Fu_names.append(os.path.join(dir_fu, item))

        print("The number of the training dataset is: {}".format(len(Fu_names)))
        if self.phase == 'train':
            x = list(enumerate(Fu_names))
            random.shuffle(x)
            indices, Fu_names = zip(*x)
            IR_names = [IR_names[idx] for idx in indices]
            VI_names = [VI_names[idx] for idx in indices]

        self.dir = None
        print("The number of the testing dataset is: {}".format(len(Fu_names)))
        self.ir_names = IR_names
        self.vi_names = VI_names
        self.Fu_names = Fu_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0] * n, [0] * n, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        ir_name = self.ir_names[index]
        vi_name = self.vi_names[index]
        fu_name = self.Fu_names[index]

        img_id = re.split('/', ir_name)[-1]
        ir_img = PIL.Image.open(ir_name).convert('RGB')
        vi_img = PIL.Image.open(vi_name).convert('RGB')
        fu_img = PIL.Image.open(fu_name).convert('RGB')

        if self.phase == 'train':
            if self.parse_patches:
                i, j, h, w = self.get_params(ir_img, (self.patch_size, self.patch_size), self.n)
                ir_img = self.n_random_crops(ir_img, i, j, h, w)
                vi_img = self.n_random_crops(vi_img, i, j, h, w)
                fu_img = self.n_random_crops(fu_img, i, j, h, w)

                ir_img = [self.transforms(ir_img[i]) for i in range(self.n)]
                vi_img = [self.transforms(vi_img[i]) for i in range(self.n)]
                fu_img = [self.transforms(fu_img[i]) for i in range(self.n)]

                outputs = {'ir':torch.stack(ir_img, dim=0),
                           'vi':torch.stack(vi_img, dim=0),
                           'fu':torch.stack(fu_img, dim=0),
                           'path':img_id}
                return outputs
            else:
                ir_img = self.transforms(ir_img)
                vi_img = self.transforms(vi_img)
                fu_img = self.transforms(fu_img)
                outputs = {'ir': ir_img.unsqueeze(0),
                           'vi': vi_img.unsqueeze(0),
                           'fu': fu_img.unsqueeze(0),
                           'path': img_id}
                return outputs
        else:
            # 在非训练阶段，直接对完整图像进行变换并拼接
            return torch.cat([self.transforms(ir_img),
                              self.transforms(vi_img),
                              ], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.Fu_names)

class Restoration:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.parse_patches = self.config['training']['parse_patches']
    def get_loaders(self, val_loaders=False):
        print("=> Utilizing the RestorationDataset() for data loading...")

        train_dataset = RestorationDataset(
            dir=os.path.join(self.config['data']['data_dir'], 'Train'),
            n=self.config['training']['patch_n'],
            patch_size=self.config['data']['image_size'],
            transforms=self.transforms,
            parse_patches=self.parse_patches,
            phase='train'
        )

        val_dataset = RestorationDataset(
            dir=os.path.join(self.config['data']['data_dir'], self.config['data']['data_name']),
            n=self.config['training']['patch_n'],
            patch_size=self.config['data']['image_size'],
            transforms=self.transforms,
            parse_patches=self.parse_patches,
            phase='val'
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        if val_loaders:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config['sampling']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=True
            )
            return train_loader, val_loader
        return train_loader, val_dataset

    def get_test_loaders(self, parse_patches=True, phase='test', data_dir=None):
        print("=> Utilizing the RestorationDataset() for data loading...")

        if data_dir is None:
            data_dir = os.path.join(self.config['data']['data_dir'])
        else:
            data_dir = os.path.join(data_dir)

        test_dataset = RestorationDataset(
            dir=data_dir,
            n=self.config['training']['patch_n'],
            patch_size=self.config['data']['image_size'],
            transforms=self.transforms,
            parse_patches=parse_patches,
            phase=phase
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['sampling']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )

        return test_loader
