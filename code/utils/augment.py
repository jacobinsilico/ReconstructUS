import torch

def add_speckle_noise(img, std=0.0001):
    noise = torch.randn_like(img) * std
    return img + img * noise

def add_jitter_and_noise(gt_tensor, contrast_factor=1.001, noise_std=0.0005, prob=1.0, seed=None):
    """
    Apply contrast jitter and optional noise to GT images.
    - Contrast is applied if contrast_factor is not None.
    - Noise is added with probability `prob`.
    - Values are clamped to [0, 1].

    Args:
        gt_tensor (torch.Tensor): [N, H, W] tensor of GT images.
        contrast_factor (float or None): Contrast scaling factor, or None to skip.
        noise_std (float): Stddev of Gaussian noise.
        prob (float): Probability to apply noise.
        seed (int or None): Random seed for reproducibility.

    Returns:
        torch.Tensor: Augmented GT tensor [N, H, W]
    """
    if seed is not None:
        torch.manual_seed(seed)

    augmented = gt_tensor.clone()

    # Contrast jitter
    if contrast_factor is not None:
        augmented = (augmented - 0.5) * contrast_factor + 0.5

    # Apply noise selectively
    if prob > 0:
        mask = torch.rand(gt_tensor.shape[0]) < prob
        noise = torch.randn_like(augmented) * noise_std
        augmented[mask] = augmented[mask] + noise[mask]

    # Clamp to [0, 1]
    augmented = torch.clamp(augmented, 0.0, 1.0)

    return augmented


import torch

def zero_out_rf_channels(rf_tensor, dropout_rate=0.1, seed=None):
    """
    Randomly zero out a fraction of transducer channels across the entire RF tensor.
    Works with both 3D [N, D, C] and 4D [B, P, D, C] tensors.

    Args:
        rf_tensor (torch.Tensor): Input RF tensor.
        dropout_rate (float): Fraction of channels to drop (e.g., 0.1 = 10%).
        seed (int or None): Optional random seed for reproducibility.

    Returns:
        torch.Tensor: Same shape as input, with dropped channels set to zero.
        list: Dropped channel indices.
    """
    if seed is not None:
        torch.manual_seed(seed)

    original_shape = rf_tensor.shape

    # Handle input shape: flatten to [N, D, C]
    if rf_tensor.ndim == 4:
        B, P, D, C = original_shape
        rf_flat = rf_tensor.view(-1, D, C)
    elif rf_tensor.ndim == 3:
        rf_flat = rf_tensor
        D, C = rf_flat.shape[1:]
    else:
        raise ValueError("rf_tensor must be 3D or 4D")

    # Select channels to zero out
    num_drop = int(C * dropout_rate)
    drop_indices = torch.randperm(C)[:num_drop].tolist()

    # Apply dropout
    rf_aug = rf_flat.clone()
    rf_aug[:, :, drop_indices] = 0.0

    # Reshape back if needed
    if rf_tensor.ndim == 4:
        rf_aug = rf_aug.view(B, P, D, C)

    return rf_aug, drop_indices