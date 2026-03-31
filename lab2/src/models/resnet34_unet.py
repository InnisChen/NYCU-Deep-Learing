import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  ResNet34 Encoder                                                    #
# ------------------------------------------------------------------ #

class BasicBlock(nn.Module):
    """ResNet34 基本 block:兩層 3x3 Conv + shortcut"""
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
#  CBAM                                                                #
# ------------------------------------------------------------------ #

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1))) * x


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


# ------------------------------------------------------------------ #
#  Bridge                                                              #
# ------------------------------------------------------------------ #

class Bridge(nn.Module):
    """Fuse layer3(256@36) + layer4(512@18) → 32@18"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(768, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )

    def forward(self, f3, f4):
        f3_down = F.interpolate(f3, size=f4.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([f3_down, f4], dim=1))


# ------------------------------------------------------------------ #
#  Decoder block                                                       #
# ------------------------------------------------------------------ #

class DecoderBlock(nn.Module):
    """Upsample → concat(skip) → Conv → ReLU → BN → CBAM"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.cbam(self.conv(x))


# ------------------------------------------------------------------ #
#  ResNet34 + UNet                                                     #
# ------------------------------------------------------------------ #

class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # ── Encoder（ResNet34）──────────────────────────────────────
        # 初始層:stride=2，輸出 286×286
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # → 143×143

        # ResNet34 層數:[3, 4, 6, 3]
        self.layer1 = _make_layer(64,  64,  num_blocks=3, stride=1)   # 143×143
        self.layer2 = _make_layer(64,  128, num_blocks=4, stride=2)   #  72×72
        self.layer3 = _make_layer(128, 256, num_blocks=6, stride=2)   #  36×36
        self.layer4 = _make_layer(256, 512, num_blocks=3, stride=2)   #  18×18

        # ── Bridge（融合 layer3 + layer4）────────────────────────────
        self.bridge = Bridge()                                           # → 32@18

        # ── Decoder（論文架構）───────────────────────────────────────
        # dec1: 32 up → concat layer4(512) → 32@36
        self.dec1 = DecoderBlock(32, skip_channels=512, out_channels=32)
        # dec2: 32 up → concat layer2(128) → 32@72
        self.dec2 = DecoderBlock(32, skip_channels=128, out_channels=32)
        # dec3: 32 up → concat layer1(64)  → 32@144
        self.dec3 = DecoderBlock(32, skip_channels=64,  out_channels=32)

        # 最後放大 × 4（144→576）再輸出，train.py center-crop → 384×384
        self.final_up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # 144→288
        self.final_up2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # 288→576
        self.out_conv  = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x  = self.init_conv(x)          # (B, 64,  286, 286)
        x  = self.maxpool(x)            # (B, 64,  143, 143)
        s1 = self.layer1(x)             # (B, 64,  143, 143)
        s2 = self.layer2(s1)            # (B, 128,  72,  72)
        f3 = self.layer3(s2)            # (B, 256,  36,  36)
        f4 = self.layer4(f3)            # (B, 512,  18,  18)

        # Bridge
        x = self.bridge(f3, f4)        # (B, 32,   18,  18)

        # Decoder
        x = self.dec1(x,  f4)          # (B, 32,   36,  36)
        x = self.dec2(x,  s2)          # (B, 32,   72,  72)
        x = self.dec3(x,  s1)          # (B, 32,  144, 144)

        x = self.final_up1(x)          # (B, 32,  288, 288)
        x = self.final_up2(x)          # (B, 32,  576, 576)
        return self.out_conv(x)        # (B, 1,   576, 576) → center-crop → 384×384


if __name__ == "__main__":
    model = ResNet34UNet()
    x = torch.randn(2, 3, 572, 572)   # actual training input (384 + 94*2 reflection pad)
    out = model(x)
    assert out.shape == (2, 1, 576, 576), f"shape error: {out.shape}"
    print("ResNet34UNet OK:", out.shape)   # (2, 1, 576, 576) → train.py center-crop → 384×384
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
