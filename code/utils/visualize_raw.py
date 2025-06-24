import torch                      
import numpy as np              
import matplotlib.pyplot as plt  


def visualize_gt_db(gt_tensor, idx=0, clim=(-60, 0)):
    """
    Visualize a ground truth image in decibel (dB) scale.

    Converts a single 2D image from a tensor batch into log-compressed dB scale
    and displays it using matplotlib.

    Args:
        gt_tensor (torch.Tensor): Tensor of shape [N, H, W] or [N, 1, H, W], ground truth images.
        idx (int): Index of the image to visualize within the batch.
        clim (tuple): Tuple of (vmin, vmax) in dB scale for color normalization.
    """
    gt = gt_tensor[idx].detach().cpu().numpy()
    gt_mag = np.abs(gt)
    gt_db = 20 * np.log10(gt_mag / np.max(gt_mag) + 1e-8)

    plt.imshow(gt_db, cmap='gray', aspect='auto', vmin=clim[0], vmax=clim[1])
    plt.title(f"GT Image #{idx} (dB scale)")
    plt.xlabel('Lateral')
    plt.ylabel('Depth')
    plt.colorbar(label='dB')
    plt.show()


def visualize_rf_line(rf_tensor, file_idx=0, pw_idx=0, ch_idx=0):
    """
    Visualize a single A-line (RF signal along depth) from an RF tensor.

    Supports both:
        - shape [N, D, C]   → N samples, D samples per channel, C channels
        - shape [N, P, D, C] → N samples, P plane waves per sample, D depth, C channels

    Args:
        rf_tensor (torch.Tensor): Input RF data (3D or 4D).
        file_idx (int): Index of the sample to visualize (axis 0).
        pw_idx (int): Index of the plane wave (only used if rf_tensor is 4D).
        ch_idx (int): Channel index to extract the A-line from.
    """
    if rf_tensor.ndim == 3:
        # Shape: [N, D, C]
        if file_idx >= rf_tensor.shape[0]:
            raise ValueError("Invalid file index for 3D tensor")
        if ch_idx >= rf_tensor.shape[2]:
            raise ValueError("Invalid channel index")
        a_line = rf_tensor[file_idx, :, ch_idx]  # [D]

    elif rf_tensor.ndim == 4:
        # Shape: [N, P, D, C]
        if file_idx >= rf_tensor.shape[0]:
            raise ValueError("Invalid file index for 4D tensor")
        if pw_idx >= rf_tensor.shape[1]:
            raise ValueError("Invalid plane wave index")
        if ch_idx >= rf_tensor.shape[3]:
            raise ValueError("Invalid channel index")
        a_line = rf_tensor[file_idx, pw_idx, :, ch_idx]  # [D]

    else:
        raise ValueError("rf_tensor must be 3D or 4D")

    # Plot
    plt.plot(a_line.cpu().numpy())
    plt.title(f"A-line (Sample #{file_idx}, PW #{pw_idx}, Channel #{ch_idx})")
    plt.xlabel("Sample (depth)")
    plt.ylabel("Amplitude (normalized)")
    plt.grid(True)
    plt.show()