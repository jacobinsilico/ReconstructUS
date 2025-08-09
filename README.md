# ReconstructUS

**ReconstructUS** is an end-to-end pipeline for ultrasound image reconstruction from raw RF data.  
It combines classical beamforming techniques, deep learning models, and FPGA-based deployment.  
This project serves as a proof-of-concept showing that high-quality ultrasound image reconstruction from a **single plane wave** is possible.

---

## Overview

The pipeline integrates:

- **MATLAB preprocessing**: Convert `.uff` files from the PICMUS/USTB datasets into `.mat` format using the Ultrasound Toolbox and perform Delay-And-Sum (DAS) reconstruction.
- **Python deep learning training**: Train U-Net and EfficientUNet models for ultrasound image reconstruction.
- **FPGA deployment**: Quantize, compile, and deploy trained models onto the **Xilinx Kria KV260 Vision AI Starter Kit** using the Vitis AI toolchain.

---

## Project Workflow

1. **Dataset acquisition**  
   Download in vivo, experimental, and simulation datasets from the [USTB dataset collection](https://www.ustb.no/ustb-datasets/).

2. **Preprocessing**  
   Use MATLAB scripts in `matlab/` to process `.uff` files into `.mat` files.  
   - The `.mat` files generated during preprocessing are **not included** in this repository.  
   - You must generate them locally using the provided scripts and the original datasets.

3. **Model training**  
   Use the provided Jupyter notebooks:
   - `train_unet.ipynb`
   - `train_effunet.ipynb`  
   These can be run in Google Colab. They use helper functions from:
   - `dataloader/` (data loading utilities)
   - `models/` (model definitions)
   - `utils/` (training and evaluation utilities)

4. **FPGA deployment**  
   Follow the `vitis-ai/` directory for the quantization, compilation, and deployment pipeline.  
   Detailed instructions are in `vitis-ai/HOWTO.md`.

---

## Repository Structure

### `dataloader/`
Data loading utilities for training and evaluation.

### `matlab/`
MATLAB scripts for preprocessing .uff files from the USTB datasets into .mat files
and performing Delay-And-Sum (DAS) reconstruction.

### `models/`
Model definitions, including U-Net and EfficientNet-based U-Net.

### `results/`
Example output images from final reconstructions.

### `utils/`
Helper functions for training, evaluation, and metric computation.

### `vitis-ai/`
Full Vitis AI deployment pipeline for the Kria KV260:
- Quantization
- Compilation
- Deployment
Includes HOWTO.md with step-by-step deployment instructions.

### `train_effunet.ipynb`
Google Colab-ready notebook to train the EfficientNet-based U-Net.

### `train_unet.ipynb`
Google Colab-ready notebook to train the U-Net.

### `requirements.txt`
Python dependencies for training and evaluation.

---

## Requirements

See `requirements.txt` for Python dependencies.  
For FPGA deployment, additional dependencies (such as `xir`) must be installed **inside the Vitis AI Docker container** — do not install them locally.

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Final Notes

If you find this repository useful, please consider starring it on GitHub or mentioning it in your work.  
At some point in the future, I may add a Linux-based workflow for Vitis AI, but for now this repository is complete.  

I hope it helps you either reproduce my results or inspires you to create something even better.

Happy computing!

---