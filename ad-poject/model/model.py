# 🔥 개선된 SwinInspiredAE (v2)
import torch
import torch.nn as nn

class SwinInspiredAEv2(nn.Module):
    def __init__(self, embed_dim=96, latent_dim=128, latent_noise_std=0.1):
        super().__init__()

        self.latent_noise_std = latent_noise_std

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # 추가된 layer
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )

        self.flatten = nn.Flatten()

        # Latent (Bottleneck)
        self.latent_fc = nn.Sequential(
            nn.Linear(embed_dim * 7 * 7, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, embed_dim * 7 * 7),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (embed_dim, 7, 7)),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(embed_dim // 2, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z_flat = self.flatten(z)
        z_latent = self.latent_fc(z_flat)

        # 🔥 Noise Injection during training
        if self.training and self.latent_noise_std > 0:
            noise = torch.randn_like(z_latent) * self.latent_noise_std
            z_latent = z_latent + noise

        x_hat = self.decoder(z_latent)
        return x_hat