import torch
import torch.nn.functional as F
import scipy.io as sio
from tqdm import tqdm

def load_gt_stack(img_paths, repeats_per_img=75, downsample_factor=1):
    """
    Loads and normalizes GT ultrasound envelope images to [0, 1] in linear scale.
    Optionally downsamples spatial resolution. Repeats each GT image to match plane waves.

    Args:
        img_paths (list): paths to .mat files
        repeats_per_img (int): how many times to repeat each GT image
        downsample_factor (int): spatial downsampling factor (e.g., 2 = half resolution)

    Returns:
        torch.Tensor: [N, H, W] in linear scale, normalized to [0, 1]
    """
    img_tensor = []
    global_max = 0.0

    # First pass: find global max
    for path in img_paths:
        img = torch.tensor(sio.loadmat(path)['img'])
        img_mag = torch.abs(img)
        global_max = max(global_max, img_mag.max().item())

    print(f"[INFO] Global max (linear magnitude): {global_max:.4f}")

    # Second pass: normalize + optionally downsample
    for path in tqdm(sorted(img_paths)):
        img = torch.tensor(sio.loadmat(path)['img'])  # complex
        img = torch.abs(img)  # envelope
        img = img / (global_max + 1e-8)  # normalize

        if downsample_factor > 1:
            img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            img = F.interpolate(img, scale_factor=1/downsample_factor, mode='bilinear', align_corners=False)
            img = img.squeeze(0).squeeze(0)  # [H, W]

            # Make sure dimensions are even
            H, W = img.shape
            if H % 2 != 0:
                img = img[:-1, :]
            if W % 2 != 0:
                img = img[:, :-1]


        for _ in range(repeats_per_img):
            img_tensor.append(img)

    return torch.stack(img_tensor)  # shape: [N, H, W]
