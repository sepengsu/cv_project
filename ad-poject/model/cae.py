import torch
import torch.nn as nn
import torch.nn.functional as F
class SlimDeepCAE_Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 128, 2), nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # ✅ Dropout 추가
            nn.Linear(64, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        noise = torch.randn_like(x) * 0.1
        x_noisy = x + noise
        encoded = self.encoder(x_noisy)
        latent = self.bottleneck(encoded)
        decoded = self.decoder(latent)
        return decoded

class SlimDeepCAE_Bottleneck32_Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 128, 2), nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.bottleneck(encoded)
        decoded = self.decoder(latent)
        return decoded
