import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import NotModel
# -----------------------------
# Discriminator
# -----------------------------
class GANDiscriminator(nn.Module, NotModel): #  이건 모델이 아니다.
    def __init__(self):
        super(GANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),   # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, 7),   # 7 -> 1x1
        )

    def forward(self, x):
        return self.main(x).view(-1)

# -----------------------------
# GANomaly (Generator + Discriminator)
# -----------------------------
class GANomaly(nn.Module):
    def __init__(self):
        super(GANomaly, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),   # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),   # 14 -> 28
            nn.Sigmoid()
        )

        # Discriminator
        self.discriminator = GANDiscriminator()

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        pred = self.discriminator(recon)
        return recon, latent, pred
