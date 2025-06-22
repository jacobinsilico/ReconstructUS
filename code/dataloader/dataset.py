import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import random

class UltrasoundDataset(Dataset):
    def __init__(self, rf_paths, img_paths, target_shape=(1, 1500, 128), augment=False):
        self.rf_paths   =  rf_paths       # paths to raw RF data [samples x transducer elements x plane waves]
        self.img_paths  =  img_paths      # paths to raw reconstructed image data [387 x 609]
        self.target_shape = target_shape  # is this needed?

        self.augment = augment  # used to perform augmentation on training GT images

    def __len__(self):
        return len(self.rf_paths)

    def __getitem__(self, idx):
        # load the raw RF data and reconstructed image
        rf   =  sio.loadmat(self.rf_paths[idx])['rf_raw']  # shape: [samples, 128, 75]
        img  =  sio.loadmat(self.img_paths[idx])['img']    # shape: [387, 609]

        # Choose one plane wave randomly
        rf = rf[:, :, random.randint(0, rf.shape[2] - 1)]  # [samples, 128]

        # Convert to tensor: shape [1, depth, transducer]
        rf   =  torch.tensor(rf, dtype=torch.float32).unsqueeze(0)
        img  =  torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Pad RF to target depth
        rf = self.pad_and_resize_rf(rf)

        # Normalize RF per-sample to [-1, 1]
        rf = self.normalize_rf(rf)

        # Normalize GT to [0, 1]
        img = img / 255.0

        # Apply augmentations if enabled
        if self.augment:
            img = self.perturb_gt(img)

        return rf, img

    def pad_and_resize_rf(self, rf):
        """Pads RF data to match target shape in depth axis if needed."""
        _, depth, width = rf.shape
        target_depth = self.target_shape[1]
        pad_amount = target_depth - depth
        if pad_amount > 0:
            rf = F.pad(rf, (0, 0, 0, pad_amount), mode='constant', value=0)
        elif pad_amount < 0:
            rf = rf[:, :target_depth, :]
        return rf

    def normalize_rf(self, rf):
        """Normalize RF sample to [-1, 1] per sample."""
        min_val = rf.min()
        max_val = rf.max()
        if max_val - min_val > 0:
            rf = 2 * (rf - min_val) / (max_val - min_val) - 1
        else:
            rf = torch.zeros_like(rf)
        return rf

    def perturb_gt(self, img):
        """Apply contrast jitter + Gaussian noise to GT image."""
        alpha = torch.empty(1).uniform_(0.95, 1.05).item()
        img = img * alpha
        img += torch.randn_like(img) * 0.005
        return torch.clamp(img, 0.0, 1.0)
