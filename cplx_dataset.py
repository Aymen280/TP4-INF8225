import numpy as np
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class cplx_SAR_dataset_npy(Dataset):
    def __init__(self, images_dir, masks_dir, size=(128, 128)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.size = size
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = np.load(img_path).astype(np.complex64)
        mask = np.load(mask_path).astype(np.float32)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        mask = (mask > 0).float()
        return image, mask