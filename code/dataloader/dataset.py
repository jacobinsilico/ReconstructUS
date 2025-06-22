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
        gt = gt.clone()
        noise = torch.randn_like(gt) * 0.000001
        gt = gt + noise
        gt = torch.clamp(gt, 0.0, 1.0)
        return gt