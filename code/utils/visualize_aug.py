import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_gt_db_aug(dataset, idx=0, clim=(-60, 0), title="GT Image (augmented, dB scale)"):
    """
    Visualizes GT image from any UltrasoundDataset-based object.
    Augmentation will be applied if dataset.augment == True.
    """
    _, gt_aug = dataset[idx]  # this calls __getitem__, applies augmentation if enabled
    gt_mag = torch.abs(gt_aug.squeeze(0)).cpu().numpy()
    gt_db = 20 * np.log10(gt_mag / np.max(gt_mag) + 1e-8)

    plt.imshow(gt_db, cmap='gray', aspect='auto', vmin=clim[0], vmax=clim[1])
    plt.title(f"{title} #{idx}")
    plt.xlabel('Lateral')
    plt.ylabel('Depth')
    plt.colorbar(label='dB')
    plt.show()


def visualize_rf_line_aug(dataset, idx=0, ch_idx=0, title_prefix="RF A-line", dropout_channels=None):
    """
    Visualizes one RF A-line from a dataset (augmented or not).
    You provide the index into dataset, and which transducer element (channel).
    Optionally, specify a list of channels that were zeroed out.
    """
    rf_aug, _ = dataset[idx]  # calls __getitem__, may zero channels if channel_dropout == True

    if ch_idx >= rf_aug.shape[1]:
        raise ValueError("Invalid transducer channel index")

    a_line = rf_aug[:, ch_idx]  # shape: [samples]

    plt.plot(a_line.cpu().numpy(), label='RF Signal')

    if dropout_channels and ch_idx in dropout_channels:
        plt.axhline(0, color='red', linestyle='--', label='Channel Zeroed')

    plt.title(f"{title_prefix} (Sample #{idx}, Channel #{ch_idx})")
    plt.xlabel("Sample (depth)")
    plt.ylabel("Amplitude (normalized)")
    if dropout_channels:
        plt.legend()
    plt.grid(True)
    plt.show()
