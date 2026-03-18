import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (ConvTranspose2d halves channels, then DoubleConv after concat)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)                  # (B, 64,  256, 256)
        s2 = self.enc2(self.pool(s1))      # (B, 128, 128, 128)
        s3 = self.enc3(self.pool(s2))      # (B, 256,  64,  64)
        s4 = self.enc4(self.pool(s3))      # (B, 512,  32,  32)

        # Bottleneck
        x = self.bottleneck(self.pool(s4)) # (B, 1024, 16,  16)

        # Decoder
        x = self.up4(x)
        x = self._pad_and_concat(x, s4)
        x = self.dec4(x)                   # (B, 512,  32,  32)

        x = self.up3(x)
        x = self._pad_and_concat(x, s3)
        x = self.dec3(x)                   # (B, 256,  64,  64)

        x = self.up2(x)
        x = self._pad_and_concat(x, s2)
        x = self.dec2(x)                   # (B, 128, 128, 128)

        x = self.up1(x)
        x = self._pad_and_concat(x, s1)
        x = self.dec1(x)                   # (B, 64,  256, 256)

        return self.out_conv(x)            # (B, 1,   256, 256)

    def _pad_and_concat(self, x, skip):
        """若尺寸不一致，將 x 插值對齊 skip 後 concat。"""
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([skip, x], dim=1)


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"shape error: {out.shape}"
    print("UNet OK:", out.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
