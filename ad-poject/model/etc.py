import torch
import torch.nn as nn
import torch.nn.functional as F

# Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        noise = torch.randn_like(x) * 0.1
        x_noisy = x + noise
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded

# Robust Autoencoder
class RobustAutoencoder(nn.Module):
    def __init__(self):
        super(RobustAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, 1, 28, 28)
        return out

# Skip Connection Autoencoder
class SkipConnectionAutoencoder(nn.Module):
    def __init__(self):
        super(SkipConnectionAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        # Decoder
        self.dec3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))

        d3 = F.relu(self.dec3(e3))
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.dec2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        out = torch.sigmoid(self.dec1(d2))
        return out