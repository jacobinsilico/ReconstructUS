import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dropout=0.2):
        super(BasicBlock, self).__init__()
        stride = 2 if downsample else 1
        self.dropout = nn.Dropout2d(p=dropout)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample or (in_channels != out_channels)
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act(out)


class CustomResNet(nn.Module):
    def __init__(self, in_channels=75, base_channels=32, output_size=128, dropout=0.2):
        super(CustomResNet, self).__init__()
        self.output_size = output_size

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)

        self.layer1 = nn.Sequential(
            BasicBlock(base_channels, base_channels, dropout=dropout),
            BasicBlock(base_channels, base_channels, dropout=dropout)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(base_channels, base_channels * 2, downsample=True, dropout=dropout),
            BasicBlock(base_channels * 2, base_channels * 2, dropout=dropout)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(base_channels * 2, base_channels * 4, downsample=True, dropout=dropout),
            BasicBlock(base_channels * 4, base_channels * 4, dropout=dropout),
            BasicBlock(base_channels * 4, base_channels * 4, dropout=dropout)
        )

        # Upsampling back to original output resolution (128×128 or 256×256)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Final projection to grayscale
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.upsample(x)
        x = self.final(x)
        x = torch.sigmoid(x)
        return F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
