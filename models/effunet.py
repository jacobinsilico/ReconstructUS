import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary 

# This model contains 970,202 trainable params and was the second one developed in the project
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ResidualSeparableBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio=2, use_dropout=False, p=0.1):
        super().__init__()
        mid_channels = out_channels // bottleneck_ratio
        self.use_projection = in_channels != out_channels

        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels),
            DepthwiseSeparableConv(mid_channels, out_channels)
        )

        self.dropout = nn.Dropout2d(p=p) if use_dropout else nn.Identity()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if self.use_projection else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x + residual


class CustomEfficientUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()

        # Encoder
        self.enc1 = ResidualSeparableBottleneckBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualSeparableBottleneckBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualSeparableBottleneckBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualSeparableBottleneckBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Decoder
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 16, base_channels * 4, kernel_size=1)
        )
        self.dec4 = ResidualSeparableBottleneckBlock(base_channels * 4 + base_channels * 8, base_channels * 4)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        )
        self.dec3 = ResidualSeparableBottleneckBlock(base_channels * 2 + base_channels * 4, base_channels * 2)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        )
        self.dec2 = ResidualSeparableBottleneckBlock(base_channels + base_channels * 2, base_channels)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, base_channels, kernel_size=1)
        )
        self.dec1 = ResidualSeparableBottleneckBlock(base_channels + base_channels, base_channels)

        self.sharpen = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x_b = self.bottleneck(self.pool4(x4))

        d4 = self.dec4(torch.cat([self.up4(x_b), x4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        out = self.sharpen(d1)
        out = self.final(out)
        return out


# --- Sanity Test ---
if __name__ == "__main__":
    model = CustomEfficientUNet(in_channels=1, base_channels=16)
    model.eval()

    x = torch.randn(1, 1, 1600, 128)
    y = model(x)
    summary(model, input_size=(1, 1, 1600, 128))
    print("Output shape:", y.shape)