import os
import numpy as np
import torch
import scipy.io as sio
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F

# -------------------------------
# 1. Normalize RF frame to [-1, 1]
# -------------------------------
def normalize_rf_frame(rf):
    min_val, max_val = rf.min(), rf.max()
    if max_val - min_val > 0:
        return 2 * (rf - min_val) / (max_val - min_val) - 1
    else:
        return torch.zeros_like(rf)

# -------------------------------
# 2. Pad RF frame to fixed depth
# -------------------------------
def pad_rf_frame(rf, target_depth):
    current_depth = rf.shape[0]
    pad_len = target_depth - current_depth
    if pad_len > 0:
        return F.pad(rf, (0, 0, 0, pad_len))  # Pad along samples axis
    else:
        return rf[:target_depth, :]  # Truncate if needed

# -------------------------------
# 3. Load + process RF tensor
# -------------------------------
def load_rf_stack(rf_paths, target_depth=1600):
    rf_tensor = []

    for path in tqdm(sorted(rf_paths)):
        rf_raw = sio.loadmat(path)['rf_raw']  # [samples, 128, 75]

        for i in range(rf_raw.shape[2]):  # loop over plane waves
            rf = rf_raw[:, :, i]  # [samples, 128]
            rf = torch.tensor(rf, dtype=torch.float32)

            rf = normalize_rf_frame(rf)
            rf = pad_rf_frame(rf, target_depth)

            rf_tensor.append(rf)

    return torch.stack(rf_tensor)  # shape: [N, target_depth, 128]

def group_plane_waves(rf_tensor, group_size=5):
    total_frames = rf_tensor.shape[0]
    usable_frames = total_frames - (total_frames % group_size)  # Drop leftovers if not divisible
    grouped = rf_tensor[:usable_frames].view(-1, group_size, rf_tensor.shape[1], rf_tensor.shape[2])
    return grouped  # shape: [B, 5, 1600, 128]