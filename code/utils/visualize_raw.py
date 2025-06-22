import os       # file access
import matplotlib.pyplot as plt     # plotting of the results (possibly later)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm   # for epoch iteration
from glob import glob   # for dataset preprocessing
from sklearn.model_selection import train_test_split # to perform validation on simulation data
import scipy.io as sio
import numpy as np

def visualize_gt_db(gt_tensor, idx=0, clim=(-60, 0)):
    gt = gt_tensor[idx].detach().cpu().numpy()
    gt_mag = np.abs(gt)
    gt_db = 20 * np.log10(gt_mag / np.max(gt_mag) + 1e-8)

    plt.imshow(gt_db, cmap='gray', aspect='auto', vmin=clim[0], vmax=clim[1])
    plt.title(f"GT Image #{idx} (dB scale)")
    plt.xlabel('Lateral')
    plt.ylabel('Depth')
    plt.colorbar(label='dB')
    plt.show()

def visualize_rf_line(rf_paths, file_idx=0, pw_idx=0, ch_idx=0):
    import scipy.io as sio

    rf_raw = sio.loadmat(rf_paths[file_idx])['rf_raw']  # shape: [samples, 128, 75]

    if pw_idx >= rf_raw.shape[2]:
        raise ValueError("Invalid plane wave index")
    if ch_idx >= rf_raw.shape[1]:
        raise ValueError("Invalid transducer channel index")

    # Extract single A-line
    a_line = rf_raw[:, ch_idx, pw_idx]  # shape: [samples]
    a_line = torch.tensor(a_line, dtype=torch.float32)

    # Normalize to [-1, 1] for visualization (optional)
    min_val, max_val = a_line.min(), a_line.max()
    if max_val - min_val > 0:
        a_line = 2 * (a_line - min_val) / (max_val - min_val) - 1

    # Plot
    plt.plot(a_line.numpy())
    plt.title(f"A-line (file #{file_idx}, PW #{pw_idx}, Channel #{ch_idx})")
    plt.xlabel("Sample (depth)")
    plt.ylabel("Amplitude (normalized)")
    plt.grid(True)
    plt.show()

