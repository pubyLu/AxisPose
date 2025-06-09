import logging

import torch
import os
from .vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from PIL import Image
import numpy as np

import torchvision.transforms as transforms

class Projection(VisionDataset):
    def __init__(self, txtPath, INPUT_H, INPUT_W,transforms = None, target_transform=None):
        super(Projection).__init__()
        self.txtPath = txtPath
        self.transform = transforms
        self.target_transform = target_transform
        self.H = INPUT_H
        self.W = INPUT_W
        self.img_path, self.mask_path, self.axis_mask_path, self.names = self.processTxt()
    def processTxt(self):
        seqs = []
        with open(self.txtPath, 'r') as file:
            seqs = file.readlines()
        img_path=[]
        mask_path=[]
        axis_mask_path=[]
        names =[]
        for seq_path in seqs:
            seq_path = seq_path.strip()
            seq_name = seq_path.split('/')[-1]
            for img in os.listdir(os.path.join(seq_path, "mask_visib")):
                name = img.split('.')[0]
                tile = img.split('.')[1]
                if tile in ["png", "jpg", "jpeg"]:
                    img_path.append(os.path.join(seq_path,'rgb',name.split('_')[0]+".png"))
                    mask_path.append(os.path.join(seq_path,'mask_visib',img))
                    axis_mask_path.append(os.path.join(seq_path, 'axis_masks',name.split('_')[0]+".png"))
                    names.append(seq_name+"_"+name)
                else:
                    continue
        return img_path, mask_path, axis_mask_path, names
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_path[idx]).convert("RGB")
            mask = Image.open(self.mask_path[idx]).convert('L')
            axis_mask = Image.open(self.axis_mask_path[idx])
            name = self.names[idx]
        except OSError as e:
            logging.error(f"Error opening image at {self.img_path[idx]}: {e}")
            # default_img = Image.new('RGB', (224, 224), (0, 0, 0))
            return None
        mask_array = np.array(mask)
        # 获取非零像素点的坐标（非零通常对应Mask中的白色部分表示物体）
        y_coords, x_coords = np.where(mask_array > 0)
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            if width< height:
                width = height
            if width > height:
                height = width

        else:
            # 如果没有找到物体（即Mask中全为0像素情况），可以进行适当的默认处理，比如设置为全0图像等，这里简单提示一下，具体按实际需求调整  ssh -p 2878 root@i-1.gpushare.com
            width, height = 0, 0
            x_min, y_min = 0, 0
        X = img.crop((x_min, y_min, x_min + width, y_min + height))
        X = X.resize((self.H, self.W), Image.BILINEAR)
        target = axis_mask.crop((x_min, y_min, x_min + width, y_min + height))
        target = target.resize((self.H, self.W),Image.BILINEAR)
        X = transforms.Compose([
            transforms.ToTensor()])(X)
        target = transforms.Compose([transforms.ToTensor()])(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target, X, name



