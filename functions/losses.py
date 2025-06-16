import math

import torch

import torch
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# 定义加载图像并转换为Tensor的函数
def load_image_as_tensor(image_path):
    """
    加载RGB图像并转换为PyTorch tensor
    """
    # 使用PIL打开图像
    image = Image.open(image_path).convert('RGB')

    # 定义转换，首先将图像转为Tensor，再归一化到[0, 1]
    transform = T.Compose([
        T.ToTensor(),  # 转换为Tensor，并且自动归一化到[0, 1]
    ])

    image_tensor = transform(image)

    return image_tensor

import copy
def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          rgb,
                          keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    rgb = rgb.to(x0.device)
    x = torch.cat([x, rgb], dim=1)
    output = model(x, t.float())

    if keepdim:
        noise_loss = (e - output).square().sum(dim=(1, 2, 3))
    else:
        noise_loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    return noise_loss


loss_registry = {
    'simple': noise_estimation_loss,
    'train': noise_estimation_loss,
}

# nohup python -u main.py --config projection.yml --exp ./exp --use_pretrained --fid --timesteps 20 --eta 1 --ni >output1.log 2>&1 &
# python main.py --config projection.yml --exp ./exp --doc Projection --sample --fid --timesteps 50 --eta 0 --ni