import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size, stride):
        super().__init__()
        self.use_residual = (in_channels == out_channels and stride == 1)
        mid_channels = in_channels * expansion

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),  # Swish replacement

            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out += x
        return out

class EfficientUNetBeamformer(nn.Module):
    def __init__(self, in_channels=75, output_size=128):
        super().__init__()
        self.output_size = output_size
        act = lambda: nn.LeakyReLU(0.1, inplace=True)

        # Encoder
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            MBConv(48, 48, expansion=1, kernel_size=3, stride=1),
            MBConv(48, 24, expansion=1, kernel_size=3, stride=1)
        )
        self.block3 = nn.Sequential(
            MBConv(24, 144, expansion=6, kernel_size=3, stride=2),
            *[MBConv(144, 240, expansion=6, kernel_size=3, stride=1) for _ in range(4)]
        )
        self.block4 = nn.Sequential(
            MBConv(240, 384, expansion=6, kernel_size=5, stride=1),
            *[MBConv(384, 384, expansion=6, kernel_size=5, stride=1) for _ in range(4)]
        )
        self.block5 = nn.Sequential(
            MBConv(384, 768, expansion=6, kernel_size=3, stride=2),
            *[MBConv(768, 768, expansion=6, kernel_size=5, stride=1) for _ in range(6)]
        )
        self.block6 = nn.Sequential(
            MBConv(768, 1056, expansion=6, kernel_size=5, stride=2),
            *[MBConv(1056, 1056, expansion=6, kernel_size=5, stride=1) for _ in range(6)]
        )
        self.block7 = nn.Sequential(
            MBConv(1056, 1824, expansion=6, kernel_size=3, stride=1),
            *[MBConv(1824, 3072, expansion=6, kernel_size=3, stride=1) for _ in range(2)]
        )

        # Decoder block helper
        def dec_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                act(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                act()
            )

        # Decoder
        self.up5 = nn.ConvTranspose2d(3072, 512, kernel_size=2, stride=2)
        self.dec5 = dec_block(512 + 1056, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = dec_block(256 + 768, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = dec_block(128 + 384, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = dec_block(64 + 240, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = dec_block(32 + 24, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.block1(x)  # [B, 48, H/2, H/2]
        x2 = self.block2(x1) # [B, 24, H/2, H/2]
        x3 = self.block3(x2) # [B, 240, H/4, H/4]
        x4 = self.block4(x3) # [B, 384, H/4, H/4]
        x5 = self.block5(x4) # [B, 768, H/8, H/8]
        x6 = self.block6(x5) # [B, 1056, H/16, H/16]
        x7 = self.block7(x6) # [B, 3072, H/16, H/16]

        # Decoder
        d5 = self.up5(x7)                      # → [B, 512, H/8]
        d5 = self.dec5(torch.cat([d5, x6], 1)) # skip: x6 = 1056

        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, x5], 1)) # skip: x5 = 768

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x4], 1)) # skip: x4 = 384

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x3], 1)) # skip: x3 = 240

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x2], 1)) # skip: x2 = 24

        out = torch.sigmoid(self.final(d1))
        return F.interpolate(out, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)

# --- Sanity test ---
if __name__ == "__main__":
    x = torch.randn(4, 75, 128, 128)
    output_size = 256
    model = EfficientUNetBeamformer(in_channels=75, output_size=output_size)
    y = model(x)
    assert y.shape == (4, 1, output_size, output_size), f"Unexpected output shape: {y.shape}"
    summary(model, input_size=(4, 75, output_size, output_size))
    print("Output shape:", y.shape)
