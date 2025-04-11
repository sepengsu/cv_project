import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepDenoisingCAE(nn.Module):
    def __init__(self, noise_std=0.1):
        super(DeepDenoisingCAE, self).__init__()
        self.noise_std = noise_std

        # ✅ Encoder: DeepCAE 그대로 (28 → 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),    # 28 → 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 14 → 7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=0),  # 7 → 5
            nn.ReLU(),
            nn.Conv2d(128, 256, 2, stride=1, padding=0), # 5 → 4
            nn.ReLU()
        )

        # ✅ Decoder: 비대칭 구조 (ConvT 2 + Conv 1, 정확히 28×28로 복원)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 32, kernel_size=5, stride=2),     # 4 → 9
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2),       # 9 → 20
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=9, stride=1, padding=4),      # 20 → 28
            nn.Sigmoid()
        )
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = torch.clamp(x + noise, 0.0, 1.0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
