import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import sys

from pytorch_nndct.apis import torch_quantizer

# Add models folder to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # /ReconstructUS/vitis-ai/quantization
models_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "models"))
sys.path.append(models_dir)

from models.effunet import CustomEfficientUNet  # models/effunet.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_dir, batch_size, subset_len):
    npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])[:subset_len]
    batches = []
    for i in range(0, len(npy_files), batch_size):
        tensors = []
        for fname in npy_files[i:i + batch_size]:
            data = np.load(os.path.join(data_dir, fname))  # [1, 1, 1600, 128]
            tensor = torch.from_numpy(data).float().squeeze(0)  # [1, 1600, 128]
            tensors.append(tensor)
        batch = torch.stack(tensors, dim=0)  # [B, 1, 1600, 128]
        batches.append(batch.to(device))
    return batches

def forward_loop(model, data_batches):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_batches, desc="Running forward loop"):
            _ = model(batch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_mode', type=str, default='calib', choices=['float', 'calib', 'test'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--subset_len', type=int, default=60)
    
    default_data_dir = os.path.join('..', "calib_images")
    parser.add_argument('--data_dir', type=str, default=default_data_dir)
    
    parser.add_argument('--deploy', action='store_true')
    parser.add_argument('--fast_finetune', action='store_true')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args()

    if args.quant_mode == 'test' and (args.batch_size != 1 or args.subset_len != 1):
        print("⚠️ For XModel export, batch size and subset_len must be 1. Adjusting...")
        args.batch_size = 1
        args.subset_len = 1

    model = CustomEfficientUNet(in_channels=1, base_channels=16).to(device)
    model.load_state_dict(torch.load(os.path.join(current_dir, "model_weights.pth"), map_location=device))
    model.eval()

    calib_input = load_data(args.data_dir, args.batch_size, args.subset_len)
    input_sample = calib_input[0]

    quantizer = torch_quantizer(
        args.quant_mode,
        model,
        (input_sample,),
        output_dir=os.path.join(current_dir, "quantized_model"),
        quant_config_file=args.config_file,
        target=args.target,
    )
    quant_model = quantizer.quant_model

    if args.fast_finetune and args.quant_mode == "calib":
        print("🔧 Running fast finetune...")
        ft_batches = load_data(args.data_dir, args.batch_size, 5120)
        quantizer.fast_finetune(forward_loop, (quant_model, ft_batches))
    elif args.fast_finetune and args.quant_mode == "test":
        print("📥 Loading fast finetuned parameters...")
        quantizer.load_ft_param()

    if args.quant_mode == "calib":
        forward_loop(quant_model, calib_input)
        quantizer.export_quant_config()
    elif args.quant_mode == "test":
        forward_loop(quant_model, calib_input)
        if args.deploy:
            quantizer.export_torch_script()
            quantizer.export_xmodel()

if __name__ == '__main__':
    main()
