import os
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob
from albumentations import Compose
import numpy as np

class CamVidDataset(Dataset):
    def __init__(self, data_dir, split="train", transform: Compose = None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.images = sorted(glob(os.path.join(data_dir, split, "images", "*.png")))
        self.masks = sorted(glob(os.path.join(data_dir, split, "masks", "*.png")))

        assert len(self.images) == len(self.masks), f"Mismatch: {len(self.images)} imgs, {len(self.masks)} masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Albumentations expects dict
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Upewniamy się, że maska to long tensor (dla CrossEntropyLoss)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        return image, mask
