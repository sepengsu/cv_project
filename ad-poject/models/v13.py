import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipNoisyDecoderAE(nn.Module):
    def __init__(self, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std

        # 🔵 Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)  # 28x28 -> 14x14
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)  # 14x14 -> 7x7
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3))  # 7x7 -> 3x3
        )

        # 🟡 Decoder (skip 없음, 단 encoder 마지막 출력만 사용)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 3 -> 6
        self.dec1 = self.decoder_block(64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 6 -> 12
        self.dec2 = self.decoder_block(32, 32)

        self.up3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # 12 -> 24
        self.dec3 = self.decoder_block(16, 16)

        self.final = nn.Conv2d(16, 1, kernel_size=3, padding=2)

    def decoder_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)  # 마지막 encoder feature

        noise = torch.randn_like(x3) * self.noise_std
        x3_noisy = x3 + noise

        d = self.up1(x3_noisy)
        d = self.dec1(d)
        d = self.up2(d)
        d = self.dec2(d)
        d = self.up3(d)
        d = self.dec3(d)

        x_hat = self.final(d)
        x_hat = F.interpolate(x_hat, size=(28, 28), mode='bilinear', align_corners=False)

        return x_hat
