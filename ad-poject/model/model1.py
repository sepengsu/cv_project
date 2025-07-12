# 4. custom model class
import torch
import torch.nn as nn
# training 방법: new0417/swin 파일 참고 또한 config는 다음과 같음 
class SwinInspiredAE(nn.Module):
    def __init__(self, embed_dim=96, latent_dim=128):
        super().__init__()

        # Encoder: simulate hierarchical patch aggregation (Swin-style)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.latent_fc = nn.Sequential(
            nn.Linear(embed_dim * 7 * 7, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, embed_dim * 7 * 7),
            nn.LeakyReLU(0.2)
        )

        # Decoder: upsampling with reversed structure
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (embed_dim, 7, 7)),
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(embed_dim // 2, 32, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)  # (B, C, 7, 7)
        z_flat = self.flatten(z)
        z_latent = self.latent_fc(z_flat)
        x_hat = self.decoder(z_latent)
        return x_hat
