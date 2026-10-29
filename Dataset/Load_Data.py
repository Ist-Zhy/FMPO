import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from natsort import natsorted


class RestorationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms, phase='train'):
        super().__init__()
        self.phase = phase
        self.transforms = transforms

        # =========================================================
        # 1. 路径配置
        # =========================================================
        if self.phase == 'train':
            # 训练模式：读取切好的 Patch
            self.patch_root = os.path.join(root_dir, 'Train')

            self.ir_dir = os.path.join(self.patch_root, 'ir')
            self.vi_dir = os.path.join(self.patch_root, 'vi')
            self.fu_dir = os.path.join(self.patch_root, 'fu')

            # 检查关键路径
            if not os.path.exists(self.ir_dir):
                raise FileNotFoundError(f"Training patches not found at {self.ir_dir}")

            # 如果 fusion GT 不存在，可以用 ir 代替
            if not os.path.exists(self.fu_dir):
                self.fu_dir = self.ir_dir

        else:
            # 验证/测试模式：读取原始大图
            self.ir_dir = os.path.join(root_dir, 'ir')
            self.vi_dir = os.path.join(root_dir, 'vi')
            self.fu_dir = os.path.join(root_dir, 'Fusion')

            if not os.path.exists(self.fu_dir):
                self.fu_dir = self.ir_dir

        # =========================================================
        # 2. 获取文件列表 (带白名单过滤功能)
        # =========================================================
        # A. 先获取所有物理存在的文件
        all_files = natsorted([f for f in os.listdir(self.ir_dir)
                               if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))])

        self.file_list = []

        # B. 训练阶段尝试加载白名单
        if self.phase == 'train':
            # 假定白名单文件放在 Train_Patches_Smart 目录下
            whitelist_name = "clean_whitelist_champion.txt"
            whitelist_path = os.path.join(self.patch_root, whitelist_name)

            # 备用路径：如果不在 Patch 目录，试试项目根目录（有时候用户忘了移进去）
            if not os.path.exists(whitelist_path):
                # 尝试在当前运行目录找
                if os.path.exists(whitelist_name):
                    whitelist_path = whitelist_name

            if os.path.exists(whitelist_path):
                print(f"[{self.phase.upper()}] 🧹 Cleaning Dataset using whitelist: {whitelist_path}")

                # 读取白名单 ID
                with open(whitelist_path, 'r') as f:
                    valid_ids = set(line.strip() for line in f if line.strip())

                # 执行过滤
                for fname in all_files:
                    fid = os.path.splitext(fname)[0]
                    if fid in valid_ids:
                        self.file_list.append(fname)

                print(f"[{self.phase.upper()}] ✅ Filtered: {len(all_files)} -> {len(self.file_list)} images.")

            else:
                print(f"[{self.phase.upper()}] ⚠️ Warning: Whitelist '{whitelist_name}' not found!")
                print(f"[{self.phase.upper()}] Using ALL {len(all_files)} images (Including potential bad data).")
                self.file_list = all_files
        else:
            # 验证/测试集通常不过滤
            self.file_list = all_files
            print(f"[{self.phase.upper()}] Dataset loaded: {len(self.file_list)} images from {self.ir_dir}")

    def __getitem__(self, index):
        name = self.file_list[index]

        # 1. 构建图片路径
        ir_path = os.path.join(self.ir_dir, name)
        vi_path = os.path.join(self.vi_dir, name)

        # GT 路径处理
        fu_path = os.path.join(self.fu_dir, name)
        if not os.path.exists(fu_path):
            fu_path = ir_path

        # 2. 读取图片 (RGB模式)
        try:
            ir_img = Image.open(ir_path).convert('RGB')
            vi_img = Image.open(vi_path).convert('RGB')
            fu_img = Image.open(fu_path).convert('RGB')
        except OSError as e:
            print(f"Error loading image: {name}")
            raise e

        # 3. 应用变换 (ToTensor, Normalize)
        if self.transforms:
            ir_t = self.transforms(ir_img)
            vi_t = self.transforms(vi_img)
            fu_t = self.transforms(fu_img)

        # 4. 返回
        return {
            'ir': ir_t,
            'vi': vi_t,
            'fu': fu_t,
            'path': name
        }

    def __len__(self):
        return len(self.file_list)


class Restoration:
    def __init__(self, config):
        self.config = config
        # 基础变换
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 根据需要开启
        ])

    def get_loaders(self, val_loaders=False):
        print("=> Initializing Loaders...")
        data_root = self.config['data']['data_dir']

        # --- 训练集 ---
        train_dataset = RestorationDataset(
            root_dir=data_root,
            transforms=self.transforms,
            phase='train'
        )

        # --- 验证集 ---
        val_root = os.path.join(data_root, 'Val')
        if not os.path.exists(val_root):
            val_root = data_root

        val_dataset = RestorationDataset(
            root_dir=val_root,
            transforms=self.transforms,
            phase='val'
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        if val_loaders:
            return train_loader, val_loader
        return train_loader, val_dataset

    def get_test_loaders(self, phase='test', data_dir=None):
        if data_dir is None:
            data_dir = self.config['data']['data_dir']

        test_dataset = RestorationDataset(
            root_dir=data_dir,
            transforms=self.transforms,
            phase=phase
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        return test_loader