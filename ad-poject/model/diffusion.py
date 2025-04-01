# diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDDPM(nn.Module):
    def __init__(self, beta_start=1e-4, beta_end=0.02, T=1000):
        super(SimpleDDPM, self).__init__()
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # U-Net 또는 간단한 CNN 가능 (여기선 간단화)
        self.denoise_fn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1).to(x0.device)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return xt, noise

    def reverse_denoise(self, xt, t):
        # xt에서 denoise_fn으로 noise 예측
        noise_pred = self.denoise_fn(xt)
        return noise_pred

    def forward(self, x0, t):
        xt, noise = self.forward_diffusion(x0, t)
        noise_pred = self.reverse_denoise(xt, t)
        return noise_pred, noise

    def loss(self, noise_pred, noise):
        return F.mse_loss(noise_pred, noise)

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(DiffusionUNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, in_channels, 3, padding=1)
        )

    def forward(self, x, t=None):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connection
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        out = self.dec1(d2)

        return out
