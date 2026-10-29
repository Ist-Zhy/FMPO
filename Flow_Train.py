import os
import yaml
from Model.Flow_MMIF import UnetVelocity
from Model.Stage1_Flow import Stage1FlowMatching
from Model.Trainer_MMIF import Trainer

import argparse

def load_config(config_path):
    """
    加载 YAML 配置文件
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def initialize_model_and_diffusion(diffusion_params):
    """
    根据配置初始化模型和扩散过程
    """
    # 创建模型
    model = UnetVelocity(
        dim=diffusion_params.get('dim', 32),
        dim_mults=(1, 1, 1, 1),
        condition=diffusion_params['condition']
    )
    # 创建扩散过程
    diffusion = Stage1FlowMatching(
        model,  # 这是前面实例化好的 UnetVelocity 模型
        image_size=diffusion_params['image_size'],
        timesteps=1000,
        sampling_timesteps=diffusion_params['sampling_timesteps'],
        loss_type=diffusion_params['loss_type'],
        condition=diffusion_params['condition']
    )
    return diffusion

def initialize_trainer(diffusion, dataset_opt, train_params):

    trainer = Trainer(
        diffusion,
        dataset=dataset_opt,
        num_samples=train_params['num_samples'],
        train_lr=8e-5,
        train_num_steps=train_params['train_num_steps'],
        gradient_accumulate_every=train_params['gradient_accumulate_every'],
        ema_decay=train_params['ema_decay'],
        amp=train_params['amp'],
        condition=train_params['condition'],
        results_folder=train_params['results_folder'],
        Experiment_folder=train_params['Experiment_folder'],
        save_and_sample_every=train_params['save_and_sample_every'],
    )

    return trainer

def main(config_path="config.yaml"):
    """
    主函数，执行训练流程
    """
    # 1. 加载配置
    config = load_config(config_path)
    # 2. 获取训练参数、扩散过程参数和数据集配置

    diffusion_params = config['Diffusion']
    train_params = config['Train_parameters']
    dataset_opt = config['Dataset_Configs']
    # autoencoder = config['autoencoder']
    # 3. 初始化模型和扩散过程
    diffusion = initialize_model_and_diffusion(diffusion_params)
    # 4. 初始化训练器
    trainer = initialize_trainer(diffusion, dataset_opt, train_params)
    # 5. 如果是主进程，加载模型并开始训练
    trainer.train()

if __name__ == "__main__":
    # 启动训练过程
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/Flow_Train.yaml")
    main(config_path=parser.parse_args().config)