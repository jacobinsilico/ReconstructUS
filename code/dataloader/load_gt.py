import torch
import scipy.io as sio
from tqdm import tqdm

def load_img_stack(img_paths, repeats_per_img=75):
    """
    Loads and normalizes GT ultrasound images, repeating each image `repeats_per_img` times.
    
    Args:
        img_paths (list of str): Paths to .mat files containing GT images (field name: 'img').
        repeats_per_img (int): Number of times to repeat each GT image to match plane waves.

    Returns:
        torch.Tensor: Tensor of shape [num_total_images, 378, 609] in range [0, 1].
    """
    img_tensor = []

    for path in tqdm(sorted(img_paths)):
        img = sio.loadmat(path)['img']  # expected shape: [378, 609]
        img = torch.tensor(img, dtype=torch.float32) / 255.0  # normalize to [0, 1]

        # Repeat for each plane wave
        for _ in range(repeats_per_img):
            img_tensor.append(img)

    return torch.stack(img_tensor)  # final shape: [450, 378, 609]
