import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CustomUNet(nn.Module):
    def __init__(self, in_channels=75, base_channels=32, output_size=128, dropout=0.2):
        super(CustomUNet, self).__init__()
        self.output_size = output_size

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8, dropout)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, dropout)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, dropout)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels, dropout)

        # Final output
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoding
        x1 = self.enc1(x)      # (B, 32, H, W)
        x2 = self.enc2(self.pool1(x1))  # (B, 64, H/2, W/2)
        x3 = self.enc3(self.pool2(x2))  # (B, 128, H/4, W/4)

        # Bottleneck
        x_b = self.bottleneck(self.pool3(x3))  # (B, 256, H/8, W/8)

        # Decoding
        d3 = self.up3(x_b)              # (B, 128, H/4, W/4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))  # skip conn

        d2 = self.up2(d3)               # (B, 64, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)               # (B, 32, H, W)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        out = self.final(d1)           # (B, 1, H, W)
        out = torch.sigmoid(out)
        return F.interpolate(out, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)