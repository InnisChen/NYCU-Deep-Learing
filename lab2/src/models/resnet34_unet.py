import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  ResNet34 Encoder                                                    #
# ------------------------------------------------------------------ #

class BasicBlock(nn.Module):
    """ResNet34 基本 block：兩層 3×3 Conv + shortcut"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(in_channels, out_channels, num_blocks, stride=1):
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ #
#  Decoder block                                                       #
# ------------------------------------------------------------------ #

class DecoderBlock(nn.Module):
    """Upsample → concat(skip) → Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ------------------------------------------------------------------ #
#  ResNet34 + UNet                                                     #
# ------------------------------------------------------------------ #

class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # ── Encoder（ResNet34）──────────────────────────────────────
        # 初始層：stride=2，輸出 128×128
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # → 64×64

        # ResNet34 層數：[3, 4, 6, 3]
        self.layer1 = _make_layer(64,  64,  num_blocks=3, stride=1)   # 64×64
        self.layer2 = _make_layer(64,  128, num_blocks=4, stride=2)   # 32×32
        self.layer3 = _make_layer(128, 256, num_blocks=6, stride=2)   # 16×16
        self.layer4 = _make_layer(256, 512, num_blocks=3, stride=2)   #  8×8

        # ── Decoder（UNet 風格）──────────────────────────────────────
        # dec3: 512 up → concat skip3(256) → 256
        self.dec3 = DecoderBlock(512,  skip_channels=256, out_channels=256)
        # dec2: 256 up → concat skip2(128) → 128
        self.dec2 = DecoderBlock(256,  skip_channels=128, out_channels=128)
        # dec1: 128 up → concat skip1(64)  → 64
        self.dec1 = DecoderBlock(128,  skip_channels=64,  out_channels=64)
        # dec0: 64  up → concat skip0(64)  → 64
        self.dec0 = DecoderBlock(64,   skip_channels=64,  out_channels=64)

        # 最後放大 × 2（128→256）再輸出
        self.final_up  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.out_conv  = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s0 = self.init_conv(x)          # (B, 64,  128, 128)
        x  = self.maxpool(s0)           # (B, 64,   64,  64)
        s1 = self.layer1(x)             # (B, 64,   64,  64)
        s2 = self.layer2(s1)            # (B, 128,  32,  32)
        s3 = self.layer3(s2)            # (B, 256,  16,  16)
        x  = self.layer4(s3)            # (B, 512,   8,   8)

        # Decoder
        x = self.dec3(x,  s3)          # (B, 256,  16,  16)
        x = self.dec2(x,  s2)          # (B, 128,  32,  32)
        x = self.dec1(x,  s1)          # (B, 64,   64,  64)
        x = self.dec0(x,  s0)          # (B, 64,  128, 128)

        x = self.final_up(x)           # (B, 32,  256, 256)
        return self.out_conv(x)        # (B, 1,   256, 256)


if __name__ == "__main__":
    model = ResNet34UNet()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"shape error: {out.shape}"
    print("ResNet34UNet OK:", out.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
