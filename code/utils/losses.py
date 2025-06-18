import torch
import torch.nn.functional as F
from piq import SSIMLoss  # SSIM from piq

# Instantiate the SSIM loss function globally (reuse it)
ssim_fn = SSIMLoss(data_range=1.0)

def ssim_loss(pred, target):
    """Returns SSIM loss using piq."""
    return ssim_fn(pred, target)

def mae_loss(pred, target):
    """Mean Absolute Error (L1) loss."""
    return F.l1_loss(pred, target)

def combined_loss(pred, target, alpha=0.84):
    """
    Weighted combination of SSIM and MAE losses.
    alpha = weight for SSIM, (1-alpha) = weight for MAE.
    """
    ssim_component = ssim_loss(pred, target)
    mae_component = mae_loss(pred, target)
    return alpha * ssim_component + (1 - alpha) * mae_component
