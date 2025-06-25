import torch

def add_speckle_noise(img, std=0.0005):
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
