# Vitis AI Deployment Pipeline (Kria KV260)

This directory documents the full deployment pipeline of a custom U-Net-based model used for **ultrasound image reconstruction** from a **single plane wave input**. The model was trained on a dataset of **coherent plane wave compounded acquisitions**, and is deployed onto the **Kria KV260 Vision AI Starter Kit** using the Vitis AI toolchain.

---

## Directory Overview

This directory is organized into three main subcomponents:

### `quantization/`
Contains all scripts required to quantize the trained PyTorch model using `vai_q_pytorch`. This includes export scripts, calibration helpers, and configuration files.

### `compilation/`
Includes the compilation script for generating the `.xmodel` file using `vai_c_xir`, targeting the KV260 DPU architecture (`DPUCZDX8G`).

### `deployment/`
Holds all necessary files to deploy and run inference on the Kria board. This includes:
- The final `.xmodel` file
- Sample input/output `.npy` data
- Python VART-based inference script (`inference.py`)
- Minimal instructions to execute inference inside the `xilinx/smartcam` Docker container

---