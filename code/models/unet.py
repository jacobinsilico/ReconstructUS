import torch
import torch.nn as nn
import torch.nn.functional as F
# should work with 5 input channels
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, p=0.1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(p=p))
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)

class CustomUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, output_size=(192,304)):
        super().__init__()
        self.output_size = output_size
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck with dropout
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, use_dropout=True)

        # Decoder with dropout
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)
        )
        self.dec3 = ConvBlock(base_channels * 4 + base_channels * 4, base_channels * 4, use_dropout=True)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        )
        self.dec2 = ConvBlock(base_channels * 2 + base_channels * 2, base_channels * 2, use_dropout=True)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        )
        self.dec1 = ConvBlock(base_channels + base_channels, base_channels)

        # Final output with sigmoid for [0, 1] normalization
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x_b = self.bottleneck(self.pool3(x3))

        d3 = self.dec3(torch.cat([self.up3(x_b), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        out = self.final(d1)
        return nn.Sigmoid(F.interpolate(out, size=self.output_size, mode='bicubic', align_corners=False))

# Example usage
if __name__ == "__main__":
    from torchinfo import summary
    model = CustomUNet(in_channels=5, base_channels=32)
    model.eval()

    dummy_input = torch.randn(8, 5, 1600, 128)
    summary(model, input_size=(8, 5, 1600, 128))
    output = model(dummy_input)
    print("Output shape:", output.shape)