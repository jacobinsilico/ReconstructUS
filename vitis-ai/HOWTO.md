# HOWTO: Vitis AI + Kria Setup

**Author:** Jakub Dawid Szkudlarek   
**Target:** Kria KV260 FPGA using Vitis AI

---

## 1. Setting Up Vitis AI on Host PC

1. **Ensure Docker Desktop is running.**  
   Pull the image `xilinx/vitis-ai-pytorch-cpu:latest`. This works for quantization.  
   For compilation, Vitis AI 2.0 is required.

2. **Start Vitis AI PyTorch container** (run from your project directory, assuming you're on Windows and using Git Bash):

   ```bash
   winpty docker run -it --name vitis-ai-pytorch-cpu -v "$PWD":/workspace xilinx/vitis-ai-pytorch-cpu:latest
   ```

3. **List available Conda environments**:

   ```bash
   conda env list
   ```

4. **Activate the correct environment** (usually `vitis-ai-pytorch`):

   ```bash
   conda activate vitis-ai-pytorch
   ```

5. **Install bindings if it's your first time:**

   ```bash
   cd /workspace/Vitis-AI/src/vai_quantizer/vai_q_pytorch/pytorch_binding
   python3 setup.py install
   ```

6. **Run calibration in your quantization directory:**

   ```bash
   python3 quantize.py --calib
   ```

   This estimates activation ranges (post-training quantization).  
   For QAT (Quantization Aware Training), refer to official Vitis-AI docs.

7. **Generate the quantized model:**

   ```bash
   python3 quantize.py --quant_mode test --deploy
   ```

   This creates the `quantized_model/` and `quantize_result/` directories with a `_int.xmodel` file.

---

## 2. Compiling for Kria (Host PC)

Before using `vai_c_xir`, confirm you’re compiling for the correct DPU architecture.

> If you're using the Smartcam Docker image on Kria, Vitis AI 2.0 is required to match model footprint.

1. **Verify `arch.json` exists inside your Docker container.**

   Target for Kria Smartcam image:
   ```
   DPUCZDX8G_ISA1_B3136
   ```

   > Use `show_dpu` on Kria to confirm the actual `dpu_footprint`.

2. **Compile using `vai_c_xir`:**

   ```bash
   vai_c_xir \
     -x CustomEfficientUNet_int.xmodel \
     -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
     -o ./deployment \
     -n CustomEfficientUNet
   ```

   This gives you the final `.xmodel` for Kria inference.

---

## 3. Sending Files to Kria (from Host)

1. **Prepare a `deployment/` directory containing:**
   - `inference.py`
   - Final `.xmodel`
   - Test inputs

2. **Send to Kria using SCP** (replace with your IP):

   ```bash
   scp -r * ubuntu@000.000.0.0:/home/ubuntu/deployment
   ```

---

## 4. Networking Setup: Kria ↔ Host

1. **Boot Kria** (takes ~1–2 minutes).

2. **Connect via PuTTY:**
   - Interface: COM4 or COM5
   - Baud rate: 115200

3. **Edit the netplan config:**

   ```bash
   sudo vim /etc/netplan/50-cloud-init.yaml
   ```

   - Set static IPs for Kria and PC
   - Enable WiFi sharing on Windows

4. **Apply changes:**

   ```bash
   sudo netplan apply
   ```

5. **Test connection:**

   ```bash
   ping 8.8.8.8
   ping google.com
   ping <your-PC-IP>
   ```

---

## 5. Smartcam Image Setup on Kria

1. **Install firmware and runtime tools:**

   ```bash
   sudo add-apt-repository ppa:xilinx-apps
   sudo apt update
   sudo apt upgrade
   sudo apt search xlnx-firmware-kv260
   ```

   Follow:  
   [Smartcam Deployment Guide](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/smartcamera/docs/app_deployment.html)

2. **Run Docker container:**

   ```bash
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
   ```

3. **Fix xclbin link inside the container:**

   ```bash
   ln -s /lib/firmware/xilinx/kv260-smartcam/kv260-smartcam.xclbin /usr/lib/dpu.xclbin
   ```

---

## 6. Inference on Kria

1. **Enter the Smartcam Docker container.**
2. **Ensure the `.xmodel`, input data, and `inference.py` are in `/workspace`.**
3. **Run inference:**

   ```bash
   python3 inference.py
   ```

   You should now see output files.  
   Transfer them back to your PC using SCP, SD card, or USB.

---

## 7. Troubleshooting Summary

- **Quantizer fails?**  
  → Check if `pytorch_nndct` is installed in your Conda environment.

- **No network on Kria?**  
  → Check `netplan` config and ensure PC sharing is working.

- **`vai_c_xir` arch mismatch?**  
  → Confirm correct `arch.json` and Docker image.

- **Bad inference results?**  
  → Try `--fast_finetune` during quantization or use QAT.

---

**Happy computing!**  
_You now have a full Vitis AI deployment pipeline running on Kria._
