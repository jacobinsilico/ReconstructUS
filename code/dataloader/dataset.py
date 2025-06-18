import torch
import torch.nn.functional as F
import scipy.io as sio

# we first define a dataset class that will take care of data handling DONE
class UltrasoundDataset(torch.utils.data.Dataset):
    def __init__(self, rf_paths, img_paths, target_shape=(75, 128, 128)):
        self.rf_paths      =  rf_paths          # paths to RF data files
        self.img_paths     =  img_paths         # paths to img files
        self.target_shape  =  target_shape      # target shape we assume each RF file to have

    def __len__(self):
        return len(self.rf_paths)   # assume 1-to-1 correspondence between RF and images

    def __getitem__(self, idx):
        # load the RF and img data from .mat files
        rf   =  sio.loadmat(self.rf_paths[idx])['rf_raw']
        img  =  sio.loadmat(self.img_paths[idx])['img']
        
        # turn the RF and img data into tensorts
        rf   =  torch.tensor(rf, dtype=torch.float32)                 # [C, H, W] -> [plane wave, depth, transducer array elements]
        img  =  torch.tensor(img, dtype=torch.float32).unsqueeze(0)   # [1, H, W] -> [1, height, width] greyscale image

        # resize both to target shape 
        rf   =  self.resize_tensor(rf, self.target_shape)
        img  =  self.resize_tensor(img, (1, self.target_shape[1], self.target_shape[2]))

        # normalize the image to avoid problems with SSIM loss later on
        img = img / 255.0
        return rf, img

    def resize_tensor(self, tensor, target_shape):
        """Resize input tensor to target_shape."""
        return F.interpolate(tensor.unsqueeze(0), size=target_shape[1:], mode='bilinear', align_corners=False).squeeze(0)