import torch
import torch.nn as nn

# ✅ SEBlock 정의 (Lightweight Attention)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.se(x)
        return x * weight

# ✅ SwinInspiredAEv4 모델
class SwinInspiredAEv3(nn.Module):
    def __init__(self, embed_dim=96, latent_dim=128, latent_noise_std=0.1):
        super().__init__()
        self.latent_noise_std = latent_noise_std

        # ✅ Encoder with SE Attention
        self.encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),

            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),

            SEBlock(embed_dim)  # 🔥 SE attention 삽입
        )

        self.flatten = nn.Flatten()

        # ✅ Nonlinear Bottleneck with LayerNorm
        self.latent_fc = nn.Sequential(
            nn.Linear(embed_dim * 7 * 7, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(latent_dim * 2),
            nn.Dropout(0.5),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, embed_dim * 7 * 7),
            nn.LeakyReLU(0.2)
        )

        # ✅ Decoder with GroupNorm
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (embed_dim, 7, 7)),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, embed_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, embed_dim // 2),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, embed_dim // 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(embed_dim // 4, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z_flat = self.flatten(z)
        z_latent = self.latent_fc(z_flat)

        if self.training and self.latent_noise_std > 0:
            noise = torch.randn_like(z_latent) * self.latent_noise_std
            z_latent = z_latent + noise

        x_hat = self.decoder(z_latent)
        return x_hat
