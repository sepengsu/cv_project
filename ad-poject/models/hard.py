import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Hard Attention MemoryAE: top-k 기반 memory matching
class HardAttentionMemoryAE(nn.Module):
    def __init__(self, input_dim=28*28, embed_dim=128, memory_size=50, topk=5):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.topk = topk

        # Encoder: image → embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

        # Memory Bank (learnable)
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))

        # Decoder: memory embedding → image
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def forward(self, x, return_attn=False):
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, 784)
        z = self.encoder(x_flat)  # (B, D)

        # Normalize for cosine similarity
        z_norm = F.normalize(z, dim=1)          # (B, D)
        mem_norm = F.normalize(self.memory, dim=1)  # (M, D)

        # Cosine similarity and hard top-k attention
        sim = torch.matmul(z_norm, mem_norm.t())  # (B, M)
        topk_vals, topk_idx = torch.topk(sim, k=self.topk, dim=1)  # (B, k)

        attn = torch.zeros_like(sim)  # (B, M)
        attn.scatter_(1, topk_idx, topk_vals)
        attn = F.softmax(attn, dim=1)  # softmax over top-k only

        # Memory read (B, M) x (M, D) → (B, D)
        z_mem = torch.matmul(attn, self.memory)

        # Decode and reshape
        x_hat = self.decoder(z_mem).view(x.size())

        if return_attn:
            return x_hat, attn
        else:
            return x_hat
