# Vitis AI Deployment Pipeline (Kria KV260)

This directory documents the full deployment pipeline of a custom U-Net-based model used for **ultrasound image reconstruction** from a **single plane wave input**.  
The model was trained on a dataset of **coherent plane wave compounded acquisitions**, and is deployed onto the **Kria KV260 Vision AI Starter Kit** using the Vitis AI toolchain.

---

## Directory Overview

This directory is organized into three main components:

### `quant_and_compile/`
Contains all files required to quantize the model with `vai_q_pytorch` and compile it with `vai_c_xir`.  
After compilation, you will obtain a final `.xmodel` file for a specific DPU architecture that can be used to run inference directly on a DPU.

This repository targets the `DPUCZDX8G` architecture. Refer to the [AMD documentation](https://docs.amd.com/r/en-US/pg338-dpu/Introduction?tocId=3xsG16y_QFTWvAJKHbisEw) for more details.

**Important:**  
The final `.xmodel` includes a DPU fingerprint that depends on the tool version used during compilation, and it must match the fingerprint of your board.  
If the fingerprints do not match, you will encounter an error.  
To resolve this, you may need to:
- Modify the `arch.json` file within Vitis AI container before using `vai_c_xir`, or  
- Use a compatible version of the Vitis AI toolchain (for this project, quantization was performed with Vitis AI 3.5, but compilation had to be done with Vitis AI 2.5 to avoid a fingerprint mismatch).

---

### `deployment/`
Contains the final `.xmodel` file obtained after compilation, a subdirectory with test data used to verify inference on the Kria KV260 FPGA, and a Python inference script.  
These files can be used to reproduce the results obtained in this project.

---

### `calib_images/`
Contains calibration images for performing fast finetuning during quantization.  
Fast finetuning is only efficient when run on a GPU; running it on a CPU is significantly slower.
