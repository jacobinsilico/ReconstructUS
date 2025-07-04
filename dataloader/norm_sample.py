import torch

def normalize_to_01(tensor, eps=1e-6):
    """Normalize 4D tensor to [0 1] range PER SAMPLE."""
    min_val = tensor.amin(dim=(1, 2, 3), keepdim=True)
    max_val = tensor.amax(dim=(1, 2, 3), keepdim=True)
    return (tensor - min_val) / (max_val - min_val + eps)