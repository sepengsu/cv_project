import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import NotModel
class Encoder(nn.Module, NotModel):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),  # 28 → 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), # 14 → 7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module, NotModel):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),   # 14 → 28
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 7, 7)
        return self.net(x)

class Discriminator(nn.Module, NotModel):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 1)  # logits
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out, feat

# 전체 GANomaly 모델 (forward는 학습에 맞게 구성)
class GANomaly(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        _, feat_fake = self.discriminator(x_hat)
        return x_hat, z, feat_fake

    def reconstruct(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
