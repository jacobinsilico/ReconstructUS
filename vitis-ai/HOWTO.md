==============================
HOWTO: Vitis AI + Kria Setup
==============================

Author: Jakub Dawid Szkudlarek
Project: SoCDAML Mini-Project
Target: Kria KV260 FPGA using Vitis AI

-----------------------------------
1. Setting Up Vitis AI on Host PC
-----------------------------------
1. Make sure Docker Desktop is running.
   -> We need to pull `xilinx/vitis-ai-pytorch-cpu:latest`. This version works for quantization. For compilation, Vitis AI 2.0 is required.

2. Start Vitis AI PyTorch container (run from your project directory, assuming you are on Windows and are using Git Bash):

   winpty docker run -it --name vitis-ai-pytorch-cpu -v "$PWD":/workspace xilinx/vitis-ai-pytorch-cpu:latest

3. Inside the container, list available conda environments:

   conda env list

4. Activate the correct one (likely `vitis-ai-pytorch`):

   conda activate vitis-ai-pytorch

5. First-time setup (needed to enable quantization):

   - Go to: `/workspace/Vitis-AI/src/vai_quantizer/vai_q_pytorch/pytorch_binding`
   - Run: `python3 setup.py install`

6. In your quantization directory (where your model weights and `quantize.py` live), run:

   python3 quantize.py --calib

   This runs calibration and estimates activation ranges. We use post-training quantization here.
   (For QAT – quantization-aware training – refer to official Vitis-AI documentation.)

7. Generate the quantized model:

   python3 quantize.py --quant_mode test --deploy

   This will produce `quantized_model` and `quantize_result` folders containing the `_int.xmodel` file.
   We will use this file for compilation using `vai_c_xir`.

-----------------------------------
2. Compiling for Kria (Host PC)
-----------------------------------
Before using `vai_c_xir`, ensure you're compiling for the correct DPU architecture.

NOTE: If you plan to use the Smartcam Docker image on Kria, you need Vitis AI 2.0 to avoid model footprint mismatch.

1. Inside the Docker container, verify that the correct `arch.json` file exists:

   Target for Kria (Smartcam): `DPUCZDX8G_ISA1_B3136`
   (This worked for my Smartcam image. If unsure, use `show_dpu` command on Kria to inspect the `dpu_footprint`.)

2. Compile using:

   vai_c_xir \
     -x CustomEfficientUNet_int.xmodel \
     -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
     -o ./deployment \
     -n CustomEfficientUNet

   This generates the final `.xmodel` for inference on Kria.

-----------------------------------
3. Sending Files to Kria (from Host)
-----------------------------------
1. Prepare a `deployment/` directory containing:
   - `inference.py` script
   - `.xmodel` from the previous step
   - Any test input data

2. Upload using SCP (replace with your Kria IP):

   scp -r * ubuntu@000.000.0.0:/home/ubuntu/deployment

-----------------------------------
4. Networking Setup: Kria ↔ Host
-----------------------------------
This setup is essential before Docker or inference:

1. Boot Kria (takes ~1–2 minutes).
2. Connect via PuTTY:
   - Interface: COM4 or COM5
   - Baud rate: 115200

3. Edit network configuration (file may differ):

   sudo vim /etc/netplan/50-cloud-init.yaml

   - Set static IPs for both Kria and your PC
   - Enable WiFi sharing on Windows

4. Apply changes:

   sudo netplan apply

5. Test connectivity:

   ping 8.8.8.8
   ping google.com
   ping <your-PC-IP>

-----------------------------------
5. Smartcam Image Setup on Kria
-----------------------------------
Before running inference:

1. Install DPU runtime and tools:

   sudo add-apt-repository ppa:xilinx-apps  
   sudo apt update  
   sudo apt upgrade  
   sudo apt search xlnx-firmware-kv260

   Then follow the official instructions:
   https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/smartcamera/docs/app_deployment.html

2. Start Docker container (very important!):

   sudo docker run \
     --env="DISPLAY" \
     -h "xlnx-docker" \
     --env="XDG_SESSION_TYPE" \
     --net=host \
     --privileged \
     --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
     -v /tmp:/tmp \
     -v /dev:/dev \
     -v /sys:/sys \
     -v /etc/vart.conf:/etc/vart.conf \
     -v /lib/firmware/xilinx:/lib/firmware/xilinx \
     -v /run:/run \
     -v "$PWD:/workspace" \
     -w /workspace \
     -it xilinx/smartcam:2022.1 bash

3. Once inside the container, run:

   ln -s /lib/firmware/xilinx/kv260-smartcam/kv260-smartcam.xclbin /usr/lib/dpu.xclbin

   This ensures proper linkage to the DPU xclbin file required for runtime.

-----------------------------------
6. Inference on Kria
-----------------------------------
1. Inside the Smartcam Docker container.
2. Ensure the `.xmodel`, input data, and inference script are under `/workspace`.
3. Run the inference script:

   python3 inference.py

   If everything worked, you should see the outputs being saved on Kria.
   Congratulations! You can now transfer the files back to your PC and check the outputs visually (via USB, SD Card if you are on Linux or scp).

-----------------------------------
7. Troubleshooting Summary
-----------------------------------
- Quantizer fails?  
  → Check if `pytorch_nndct` is installed and activated in conda env.

- No network on Kria?  
  → Review your netplan config, PC IP, and WiFi sharing.

- `vai_c_xir` arch errors?  
  → Verify `arch.json` path and DPU target name. Use the correct Docker version.

- `.xmodel` works but output is bad?  
  → Try adding `--fast_finetune` during quantization or use QAT instead.

Happy computing!
--------------------
END OF HOWTO FILE
--------------------