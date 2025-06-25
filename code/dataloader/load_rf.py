import os
import numpy as np
import torch
import scipy.io as sio
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F

# -------------------------------
# 1. Pad RF frame to fixed depth
# -------------------------------
def pad_rf_frame(rf, target_depth):
    current_depth = rf.shape[0]
    pad_len = target_depth - current_depth
    if pad_len > 0:
        return F.pad(rf, (0, 0, 0, pad_len))  # Pad along samples axis
    else:
        return rf[:target_depth, :]  # Truncate if needed

# -------------------------------
# 2. First pass: compute global min/max
# -------------------------------
def compute_global_min_max(rf_paths):
    min_val = float('inf')
    max_val = float('-inf')

    print("Computing global min/max...")
    for path in tqdm(sorted(rf_paths)):
        rf_raw = sio.loadmat(path)['rf_raw']  # shape: [samples, 128, 75]
        min_val = min(min_val, rf_raw.min())
        max_val = max(max_val, rf_raw.max())

    return float(min_val), float(max_val)

# -------------------------------
# 3. Normalize RF frame to [-1, 1] using global min/max
# -------------------------------
def normalize_rf_frame(rf, global_min, global_max):
    if global_max - global_min > 0:
        return 2 * (rf - global_min) / (global_max - global_min) - 1
    else:
        return torch.zeros_like(rf)

# -------------------------------
# 4. Second pass: load + normalize + pad all RF frames
# -------------------------------
def load_rf_stack(rf_paths, target_depth=1600, global_min=None, global_max=None):
    rf_tensor = []

    print("Loading and normalizing RF frames...")
    for path in tqdm(sorted(rf_paths)):
        rf_raw = sio.loadmat(path)['rf_raw']  # shape: [samples, 128, 75]

        for i in range(rf_raw.shape[2]):  # loop over plane waves
            rf = rf_raw[:, :, i]  # shape: [samples, 128]
            rf = torch.tensor(rf, dtype=torch.float32)

            rf = normalize_rf_frame(rf, global_min, global_max)
            rf = pad_rf_frame(rf, target_depth)

            rf_tensor.append(rf)

    return torch.stack(rf_tensor)  # shape: [N, target_depth, 128]

# -------------------------------
# 5. Group RF frames into batches of plane waves
# -------------------------------
def group_plane_waves(rf_tensor, group_size=5):
    total_frames = rf_tensor.shape[0]
    usable_frames = total_frames - (total_frames % group_size)  # Drop leftovers if not divisible
    grouped = rf_tensor[:usable_frames].view(-1, group_size, rf_tensor.shape[1], rf_tensor.shape[2])
    return grouped  # shape: [B, 5, 1600, 128]

# -------------------------------
# 6. Example usage (if needed)
# -------------------------------
if __name__ == "__main__":
    rf_paths = sorted(glob(r"C:\Users\jdszk\OneDrive - ETH Zurich\Desktop\SoCDAML_Project\ReconstructUS\data\raw\*.mat"))

    # First pass
    global_min, global_max = compute_global_min_max(rf_paths)

    # Second pass
    rf_tensor = load_rf_stack(rf_paths, target_depth=1600, global_min=global_min, global_max=global_max)

    # Group into batches of 5 plane waves
    grouped_rf = group_plane_waves(rf_tensor, group_size=5)

    print("Grouped RF tensor shape:", grouped_rf.shape)
