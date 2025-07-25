from torch.utils.data import Dataset, ConcatDataset
import torch
import random

class UltrasoundDataset(Dataset):
    def __init__(self, rf_tensor, gt_tensor, augment=False, channel_dropout=False, dropout_rate=0.1):
        self.rf_tensor = rf_tensor
        self.gt_tensor = gt_tensor
        self.augment = augment
        self.channel_dropout = channel_dropout
        self.dropout_rate = dropout_rate

    def __len__(self):
        return self.rf_tensor.shape[0]

    def __getitem__(self, idx):
        rf = self.rf_tensor[idx].clone()  # [D, 128]
        gt = self.gt_tensor[idx].unsqueeze(0).clone()  # [1, 378, 609]

        if self.channel_dropout:
            rf = self.apply_channel_dropout(rf)

        if self.augment:
            gt = self.augment_gt(gt)

        return rf, gt

    def apply_channel_dropout(self, rf):
        num_channels = rf.shape[1]
        num_drop = int(num_channels * self.dropout_rate)
        drop_indices = random.sample(range(num_channels), num_drop)
        rf[:, drop_indices] = 0
        return rf

    def augment_gt(gt):
        # Apply contrast jitter
        contrast_factor = 1.001  # or slightly higher if you want more visual effect
        gt = (gt - 0.5) * contrast_factor + 0.5

        # Add Gaussian noise
        noise_std = 0.0001  # tune this if needed
        noise = torch.randn_like(gt) * noise_std
        gt = gt + noise

        # Normalize again to [0, 1]
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        
        return gt
