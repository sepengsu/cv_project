import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)

class UNetMemAEv14LatentPlusCentral(nn.Module):
    def __init__(self):
        super().__init__()

        # 헅 구조 인코더
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 중앙 지량 인코더
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3))
        )

        # 바트널넷
        self.to_latent = nn.Linear(128 + 576, 512)
        self.res_gate = ResidualGate(512)

        # 디코더 (f_text 추가한 입력)
        self.up1 = nn.ConvTranspose2d(512 + 576, 128, 4, stride=2, padding=1)
        self.dec1 = self.decoder_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.dec2 = self.decoder_block(64, 32)

        self.up3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.dec3 = self.decoder_block(32, 32)

        self.up4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec4 = self.decoder_block(16, 16)

        self.up5 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def crop_center(self, x, size=14):
        _, _, H, W = x.shape
        start = (H - size) // 2
        return x[:, :, start:start+size, start:start+size]

    def decoder_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
        )

    def forward(self, x, return_latent=False):
        B = x.size(0)

        # 구조 인코더
        f_struct = self.structure_encoder(x).view(B, -1)

        # 중앙 지량 인코더
        x_crop = self.crop_center(x, 14)
        f_text = self.texture_encoder(x_crop).view(B, -1)

        # 통합 latent
        latent = self.to_latent(torch.cat([f_struct, f_text], dim=1))
        f_latent = self.res_gate(latent)  # (B, 512)

        # f_text 도 decoder에 추가 전달
        d_input = torch.cat([f_latent, f_text], dim=1).view(B, 512 + 576, 1, 1)
        d = self.up1(d_input); d = self.dec1(d)
        d = self.up2(d); d = self.dec2(d)
        d = self.up3(d); d = self.dec3(d)
        d = self.up4(d); d = self.dec4(d)
        d = self.up5(d)
        x_hat = self.final(d)

        if x_hat.shape[-1] != 28:
            x_hat = F.interpolate(x_hat, size=(28, 28), mode='bilinear', align_corners=False)

        return (x_hat, f_latent) if return_latent else x_hat