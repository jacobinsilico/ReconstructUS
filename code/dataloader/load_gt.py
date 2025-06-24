import torch
import scipy.io as sio
from tqdm import tqdm

def load_gt_stack(img_paths, repeats_per_img=75):
    """
    Loads and normalizes GT ultrasound envelope images to [0, 1] in linear scale.
    Repeats each GT image to match plane waves.

    Returns:
        torch.Tensor: [N, H, W] in linear scale, normalized to [0, 1]
    """
    img_tensor = []

    # First pass: find global max across all images
    global_max = 0.0
    for path in img_paths:
        img = torch.tensor(sio.loadmat(path)['img'])
        img_mag = torch.abs(img)
        global_max = max(global_max, img_mag.max().item())

    print(f"[INFO] Global max (linear magnitude): {global_max:.4f}")

    # Second pass: normalize each image using global max
    for path in tqdm(sorted(img_paths)):
        img = torch.tensor(sio.loadmat(path)['img'])  # complex
        img = torch.abs(img)  # envelope
        img = img / (global_max + 1e-8)  # normalize to [0, 1]

        for _ in range(repeats_per_img):
            img_tensor.append(img)

    return torch.stack(img_tensor)  # shape: [N, H, W]
