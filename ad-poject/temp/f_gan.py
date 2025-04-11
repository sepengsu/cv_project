import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import NotModel
# =============================
# 4. f-AnoGAN 구성 (Generator + Encoder + Discriminator)
# =============================

class fAnoGAN_Generator(nn.Module, NotModel):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class fAnoGAN_Encoder(nn.Module, NotModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128)
        )

    def forward(self, x):
        return self.net(x)

class fAnoGAN_Discriminator(nn.Module, NotModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    

class fAnoGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = fAnoGAN_Encoder()
        self.generator = fAnoGAN_Generator()
        self.discriminator = fAnoGAN_Discriminator()

    def encode(self, x):
        return self.encoder(x)

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.generate(z)
        d_real = self.discriminate(x)
        d_fake = self.discriminate(x_hat)
        return z, x_hat, d_real, d_fake