import torch

model = torch.load("CustomEfficientUNet_int.pt")
print(model.input_quantizer)
