import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Memory-Augmented Autoencoder
class MemoryAE(nn.Module):
    def __init__(self, input_dim=28*28, embed_dim=128, memory_size=50):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.memory_size = memory_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

        # Memory Bank (learnable)
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))  # (M, D)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def forward(self, x, return_attn=False):
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, 784)
        z = self.encoder(x_flat)  # (B, D)

        # Normalize for cosine similarity
        z_norm = F.normalize(z, dim=1)  # (B, D)
        mem_norm = F.normalize(self.memory, dim=1)  # (M, D)

        # Cosine similarity → softmax attention
        sim = torch.matmul(z_norm, mem_norm.t())  # (B, M)
        attn = F.softmax(sim, dim=1)  # (B, M)

        # Attention-weighted memory read
        z_mem = torch.matmul(attn, self.memory)  # (B, D)

        # Decoder
        x_hat = self.decoder(z_mem).view(x.size())  # (B, 1, 28, 28)

        if return_attn:
            return x_hat, attn
        else:
            return x_hat
