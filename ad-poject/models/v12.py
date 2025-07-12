import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetMemAEv12Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 🔵 구조 인코더
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 → 14
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        )

        # 🔶 중앙 질감 인코더
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 → 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3))  # (B, 64, 3, 3)
        )

        # 🧠 Bottleneck: 구조(128) + 질감(64×3×3=576)
        self.to_latent = nn.Linear(128 + 576, 256)

        # 🟡 업샘플링 기반 Decoder (28x28로 확장)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 1 → 2
        self.dec1 = self.decoder_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)   # 2 → 4
        self.dec2 = self.decoder_block(64, 32)

        self.up3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)   # 4 → 8
        self.dec3 = self.decoder_block(32, 32)

        self.up4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)   # 8 → 16
        self.dec4 = self.decoder_block(16, 16)

        self.up5 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)    # 16 → 32
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def crop_center(self, x, size=14):
        _, _, H, W = x.shape
        start = (H - size) // 2
        return x[:, :, start:start+size, start:start+size]

    def decoder_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, return_latent=False):
        B = x.size(0)

        # 🔵 구조 인코딩
        f_struct = self.structure_encoder(x).view(B, -1)  # (B, 128)

        # 🔶 중앙 질감 인코딩
        x_crop = self.crop_center(x, 14)
        f_text = self.texture_encoder(x_crop).view(B, -1)  # (B, 576)

        # 🧠 통합 latent
        f_latent = self.to_latent(torch.cat([f_struct, f_text], dim=1))  # (B, 256)

        # 🟡 Decoder
        d = f_latent.view(B, 256, 1, 1)  # 1x1
        d = self.up1(d)  # 2x2
        d = self.dec1(d)

        d = self.up2(d)  # 4x4
        d = self.dec2(d)

        d = self.up3(d)  # 8x8
        d = self.dec3(d)

        d = self.up4(d)  # 16x16
        d = self.dec4(d)

        d = self.up5(d)  # 32x32
        x_hat = self.final(d)  # (B, 1, 32, 32)

        # 🔧 Interpolate → 28x28 맞추기
        if x_hat.shape[-1] != 28:
            x_hat = F.interpolate(x_hat, size=(28, 28), mode='bilinear', align_corners=False)

        return (x_hat, f_latent) if return_latent else x_hat
