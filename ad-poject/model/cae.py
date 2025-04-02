import torch
import torch.nn as nn
import torch.nn.functional as F

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 14 -> 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class DeepCAE(nn.Module):
    def __init__(self):
        super(DeepCAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=0), # 7 -> 5
            nn.ReLU(),
            nn.Conv2d(128, 256, 2, stride=1, padding=0),# 5 -> 4
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=1),       # 4 -> 5
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=1),        # 5 -> 7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 14 -> 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class HybridCAE(nn.Module):
    def __init__(self):
        super(HybridCAE, self).__init__()
        # CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
        )
        # ANN Bottleneck
        self.fc_enc = nn.Sequential(
            nn.Flatten(),  # 64 x 7 x 7 -> 3136
            nn.Linear(64*7*7, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_dec = nn.Sequential(
            nn.Linear(256, 64*7*7),
            nn.ReLU(),
        )
        # CNN Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 14 -> 28
            nn.Sigmoid()
        )

    def forward(self, x):
        enc_feat = self.encoder(x)              # [B, 64, 7, 7]
        z = self.fc_enc(enc_feat)               # [B, 256]
        z = self.fc_dec(z).view(-1, 64, 7, 7)  # [B, 64, 7, 7]
        out = self.decoder(z)                   # [B, 1, 28, 28]
        return out
