import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_metrics(y_true, y_pred):
    """
    Computes PSNR and SSIM between two numpy arrays.

    Args:
        y_true (np.ndarray): Ground truth image, shape (H, W) or (1, H, W)
        y_pred (np.ndarray): Predicted image, shape (H, W) or (1, H, W)

    Returns:
        tuple: (PSNR, SSIM) float values
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    assert y_true.ndim == 2, f"Expected 2D arrays after squeeze, got {y_true.ndim}D"

    psnr_val = psnr(y_true, y_pred, data_range=1.0)
    ssim_val = ssim(y_true, y_pred, data_range=1.0)

    return psnr_val, ssim_val
