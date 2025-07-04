import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, p=0.1):
        super().__init__()
        self.use_projection = in_channels != out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dropout = nn.Dropout2d(p=p) if use_dropout else nn.Identity()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if self.use_projection else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x + residual

class CustomUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, output_size=(192, 304)):
        super().__init__()
        self.output_size = output_size

        # Encoder
        self.enc1 = ResidualConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck with dilated conv
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Decoder
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)
        )
        self.dec3 = ResidualConvBlock(base_channels * 4 + base_channels * 4, base_channels * 4)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        )
        self.dec2 = ResidualConvBlock(base_channels * 2 + base_channels * 2, base_channels * 2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        )
        self.dec1 = ResidualConvBlock(base_channels + base_channels, base_channels)

        # Final sharpening and output layer
        self.sharpen = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x_b = self.bottleneck(self.pool3(x3))

        d3 = self.dec3(torch.cat([self.up3(x_b), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        out = self.sharpen(d1)
        out = self.final(out)

        return F.interpolate(out, size=self.output_size, mode='bicubic', align_corners=False)

# Example usage
if __name__ == "__main__":
    from torchinfo import summary
    model = CustomUNet(in_channels=1, base_channels=16)
    model.eval()

    dummy_input = torch.randn(1, 1, 1600, 128)
    summary(model, input_size=(1, 1, 1600, 128))
    output = model(dummy_input)
    print("Output shape:", output.shape)