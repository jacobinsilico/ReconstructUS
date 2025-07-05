import torch
import sys
import os

# Set proper paths relative to current file (/ReconstructUS/vitis-ai/quantization)
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(models_dir)

from models.unet import CustomUNet
from models.effunet import CustomEfficientUNet

# Configuration
dummy_input = torch.randn(1, 1, 1600, 128)  # adjust if needed
output_dir = os.path.join(current_dir, "trained_unet")  

# Export UNet
unet = CustomUNet(in_channels=1, base_channels=24)  # Update args if needed
unet.load_state_dict(torch.load(os.path.join(output_dir, "model_weights.pth"), map_location="cpu"))
unet.eval()

torch.onnx.export(
    unet,
    dummy_input,
    os.path.join(output_dir, "unet.onnx"),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

# Export EfficientUNet
output_dir = os.path.join(current_dir, "trained_effunet")

effnet = CustomEfficientUNet(in_channels=1, base_channels=16)  # Update args if needed
effnet.load_state_dict(torch.load(os.path.join(output_dir, "model_weights.pth"), map_location="cpu"))
effnet.eval()

torch.onnx.export(
    effnet,
    dummy_input,
    os.path.join(output_dir, "effnet.onnx"),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("Export complete: UNet and EfficientUNet saved to ONNX.")
