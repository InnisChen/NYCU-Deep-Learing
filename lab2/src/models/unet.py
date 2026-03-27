import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv2d → ReLU → Conv2d → ReLU  (valid conv, no padding, no BN)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    UNet following arXiv:1505.04597 with valid convolutions.
    Adapted for 256×256 input (paper uses 572×572).
    Skip connections use center-crop to match decoder spatial size.
    Output channels = 1 for binary segmentation.

    Spatial trace (256×256 input):
        enc1: 252  pool→126  enc2: 122  pool→61  enc3: 57
        pool→28    enc4: 24  pool→12   bottleneck: 8
        up→16  dec4: 12  up→24  dec3: 20  up→40  dec2: 36  up→72  dec1: 68
        output: (B, 1, 68, 68)
    """
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

        # Decoder: ConvTranspose2d upsamples, then DoubleConv after crop+concat
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 1×1 output conv
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _crop_and_concat(self, encoder_feat, decoder_feat):
        """Center-crop encoder_feat to match decoder_feat's H×W, then concat."""
        _, _, eH, eW = encoder_feat.shape
        _, _, dH, dW = decoder_feat.shape
        crop_h = (eH - dH) // 2
        crop_w = (eW - dW) // 2
        encoder_feat = encoder_feat[:, :, crop_h:crop_h + dH, crop_w:crop_w + dW]
        return torch.cat([encoder_feat, decoder_feat], dim=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)               # (B,  64, 252, 252)
        s2 = self.enc2(self.pool(s1))   # (B, 128, 122, 122)
        s3 = self.enc3(self.pool(s2))   # (B, 256,  57,  57)
        s4 = self.enc4(self.pool(s3))   # (B, 512,  24,  24)

        # Bottleneck
        x = self.bottleneck(self.pool(s4))  # (B, 1024,  8,   8)

        # Decoder
        x = self.up4(x)                     # (B,  512, 16,  16)
        x = self._crop_and_concat(s4, x)    # crop s4(24→16), cat → (B, 1024, 16, 16)
        x = self.dec4(x)                    # (B,  512, 12,  12)

        x = self.up3(x)                     # (B,  256, 24,  24)
        x = self._crop_and_concat(s3, x)    # crop s3(57→24), cat → (B,  512, 24, 24)
        x = self.dec3(x)                    # (B,  256, 20,  20)

        x = self.up2(x)                     # (B,  128, 40,  40)
        x = self._crop_and_concat(s2, x)    # crop s2(122→40), cat → (B, 256, 40, 40)
        x = self.dec2(x)                    # (B,  128, 36,  36)

        x = self.up1(x)                     # (B,   64, 72,  72)
        x = self._crop_and_concat(s1, x)    # crop s1(252→72), cat → (B, 128, 72, 72)
        x = self.dec1(x)                    # (B,   64, 68,  68)

        return self.out_conv(x)             # (B,    1, 68,  68)


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Output shape: {out.shape}")  # expected: (2, 1, 68, 68)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
