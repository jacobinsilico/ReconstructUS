import torch
import torch.nn as nn
import torch.nn.functional as F
# try this, 969 k params
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.op(x)

class MicroUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 4 + base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2 + base_channels * 2, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels + base_channels, base_channels)

        # Final layer
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        # Bottleneck
        x_b = self.bottleneck(self.pool3(x3))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(x_b), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        out = self.final(d1)
        out = torch.sigmoid(out)
        return F.interpolate(out, size=(387, 609), mode='bicubic', align_corners=False)

# Example usage
if __name__ == "__main__":
    from torchinfo import summary

    model = MicroUNet(in_channels=1, base_channels=32)
    model.eval()

    dummy_input = torch.randn(1, 1, 1800, 128)  # Example: batch size 8
    summary(model, input_size=(1, 1, 1800, 128))
    output = model(dummy_input)
    print("Output shape:", output.shape)