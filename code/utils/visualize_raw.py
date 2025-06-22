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
    #gt = torch.clamp(gt_tensor[idx], 0.0, 1.0).detach().cpu().numpy()
    gt_mag = np.abs(gt)
    gt_db = 20 * np.log10(gt_mag / np.max(gt_mag) + 1e-8)

    plt.imshow(gt_db, cmap='gray', aspect='auto', vmin=clim[0], vmax=clim[1])
    plt.title(f"GT Image #{idx} (dB scale)")
    plt.xlabel('Lateral')
    plt.ylabel('Depth')
    plt.colorbar(label='dB')
    plt.show()

import torch
import matplotlib.pyplot as plt

def visualize_rf_line(rf_tensor, file_idx=0, pw_idx=0, ch_idx=0):
    """
    Visualize a single RF A-line from a given rf_tensor [N, D, C].

    Args:
        rf_tensor (torch.Tensor): shape [N, D, C]
        file_idx (int): which RF sample to visualize
        pw_idx (int): not used (for legacy compatibility)
        ch_idx (int): which transducer channel to plot
    """
    if file_idx >= rf_tensor.shape[0]:
        raise ValueError("Invalid file index")
    if ch_idx >= rf_tensor.shape[2]:
        raise ValueError("Invalid channel index")

    a_line = rf_tensor[file_idx, :, ch_idx]  # shape: [depth]
    
    # Plot
    plt.plot(a_line.cpu().numpy())
    plt.title(f"A-line (Sample #{file_idx}, Channel #{ch_idx})")
    plt.xlabel("Sample (depth)")
    plt.ylabel("Amplitude (normalized)")
    plt.grid(True)
    plt.show()


