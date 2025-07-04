import torch
import torch.nn.functional as F
from piq import SSIMLoss           # optional: piq-based SSIM
from pytorch_msssim import ms_ssim  # MS-SSIM
import torch.nn as nn

# === 1. SSIM Loss (piq version) ===
ssim_fn = SSIMLoss(data_range=1.0)

def ssim_loss(pred, target):
    """SSIM loss using piq (single-scale)."""
    return ssim_fn(pred, target)

# === 2. MS-SSIM Loss ===
def ms_ssim_loss(pred, target):
    """Multi-scale SSIM loss."""
    return 1 - ms_ssim(pred, target, data_range=1.0, size_average=True)

# === 3. MAE / L1 Loss ===
def mae_loss(pred, target):
    return F.l1_loss(pred, target)

# === 4. Charbonnier Loss ===
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

# === 5. Edge (Sobel) Loss ===
def sobel_edges(x):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    edge_x = F.conv2d(x, sobel_x, padding=1)
    edge_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

def edge_loss(pred, target):
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))

# === 6. FFT Loss ===
def fft_loss(pred, target):
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    return torch.mean(torch.abs(pred_fft - target_fft))