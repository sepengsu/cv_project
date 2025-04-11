import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepUpsampleCAE(nn.Module):
    def __init__(self):
        super(DeepUpsampleCAE, self).__init__()

        # 🔸 인코더 (그대로 유지: 깊게 압축)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),    # 28 → 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 14 → 7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),             # 7 → 5
            nn.ReLU(),
            nn.Conv2d(128, 256, 2, stride=1),            # 5 → 4
            nn.ReLU()
        )

        # 🔸 디코더 (더 단순화)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4 → 8
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),                 # 8 → 16
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),                 # 16 → 32
            nn.Conv2d(64, 1, 9, padding=2),              # 32 → 28 (crop or Conv 후에 맞춤)
            nn.Sigmoid()
        )





    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
