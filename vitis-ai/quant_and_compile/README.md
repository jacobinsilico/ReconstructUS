# Quantization and Compilation

This directory contains all files and scripts needed to **quantize** the trained model using `vai_q_pytorch` and then **compile** it using `vai_c_xir` for deployment on the Kria KV260 FPGA.

---

## Directory Structure

### `trained_effunet/`
Contains the pre-trained "efficient" U-Net model used in this project.  
Includes:
- `effnet.onnx` – ONNX export of the EfficientUNet model.
- `model_weights.pth` – PyTorch weights.
- `training_state.pth` – Training state (optimizer, epoch, etc.).

### `trained_unet/`
Contains the pre-trained U-Net model used in this project.  
Includes:
- `unet.onnx` – ONNX export of the U-Net model.
- `model_weights.pth` – PyTorch weights.
- `training_state.pth` – Training state (optimizer, epoch, etc.).

These two folders allow you to reproduce the results without retraining the models.

---

### `quantized_model/`
Intermediate model representation generated during quantization with the `python_nndct` tool (part of the Vitis AI toolchain).  
Includes:
- `CustomEfficientUNet.py` – Model definition adapted for quantization.
- `bias_corr.pth` – Bias correction file produced by quantization.
- `quant_info.json` – Quantization metadata.

---

### `quantize_result/`
Contains the outputs of the quantization step:
- `CustomEfficientUNet_int.pt` – Quantized PyTorch model.
- `CustomEfficientUNet_int.xmodel` – Quantized `.xmodel` (before compilation).
- `check_model_data.py` – Utility script to inspect the quantized model.

#### `quantize_result/deployment/`
Final compiled `.xmodel` along with other files produced by `vai_c_xir`, ready for deployment to the DPU on the Kria KV260. It is the same `.xmodel` as the one in `vitis-ai/deployment`.

---

### `export_to_onnx.py`
Script to export the trained PyTorch model to the ONNX format.

### `quantize.py`
Script to run the quantization process. Can be used with or without calibration images.

---