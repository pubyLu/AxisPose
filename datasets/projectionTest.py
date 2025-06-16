import logging

import torch
import os
from .vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from PIL import Image
import numpy as np

import torchvision.transforms as transforms

class ProjectionTest(VisionDataset):
    def __init__(self, txtPath, INPUT_H, INPUT_W,transforms = None, target_transform=None):
        super(ProjectionTest).__init__()
        self.txtPath = txtPath
        self.transform = transforms
        self.target_transform = target_transform
        self.H = INPUT_H
        self.W = INPUT_W
        self.img_path, self.axis_mask_path, self.names = self.processTxt()
    def processTxt(self):
        seqs = []
        with open(self.txtPath, 'r') as file:
            seqs = file.readlines()
        img_path=[]
        axis_mask_path=[]
        names = []
        # /hy-tmp/dataset/train_/train/01124
        for seq_path in seqs:
            seq_path = seq_path.strip()
            for img in os.listdir(os.path.join(seq_path, "masked_rgb")):
                name=img.split('.')[0]
                names.append(name)
                img_path.append(os.path.join(seq_path,'masked_rgb',img))
                axis_mask_path.append(os.path.join(seq_path, 'axis_masks',img))
        return img_path, axis_mask_path,names
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_path[idx]).convert("RGB")
            axis_mask = Image.open(self.axis_mask_path[idx])
        except OSError as e:
            logging.error(f"Error opening image at {self.img_path[idx]}: {e}")
            # default_img = Image.new('RGB', (224, 224), (0, 0, 0))
            return None
        X = img.resize((self.H, self.W), Image.BILINEAR)
        target = axis_mask.resize((self.H, self.W),Image.BILINEAR)
        X = transforms.Compose([
            transforms.ToTensor()])(X)
        target = transforms.Compose([transforms.ToTensor()])(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target, X, self.names[idx]



