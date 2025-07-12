import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=32, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, heads=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x

class MemoryModule(nn.Module):
    def __init__(self, mem_dim=256, mem_size=100, topk=10):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, mem_dim))
        self.topk = topk

    def forward(self, x):
        mem_norm = F.normalize(self.memory, dim=1)
        x_norm = F.normalize(x, dim=1)
        sim = torch.matmul(x_norm, mem_norm.T)
        topk_sim, topk_idx = torch.topk(sim, self.topk, dim=1)
        weights = F.softmax(topk_sim, dim=1)
        selected_mem = self.memory[topk_idx]
        recon = torch.sum(weights.unsqueeze(-1) * selected_mem, dim=1)
        return recon

class SwinMemAE(nn.Module):
    def __init__(self, embed_dim=32, patch_size=4, memory_size=100):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=1, embed_dim=embed_dim, patch_size=patch_size)
        self.encoder = nn.Sequential(
            SwinBlock(embed_dim),
            SwinBlock(embed_dim)
        )
        self.to_latent = nn.Linear(embed_dim * 49, 256)
        self.memory = MemoryModule(mem_dim=256, mem_size=memory_size, topk=10)

        # Decoder with dropout and reduced capacity
        self.dropout = nn.Dropout(p=0.3)
        self.dec_fc = nn.Linear(256, 64 * 7 * 7)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )

    def forward(self, x, return_latent=False):
        B = x.size(0)
        x_patch = self.patch_embed(x)
        feat = self.encoder(x_patch)
        flat = feat.reshape(B, -1)
        latent = self.to_latent(flat)
        mem_latent = self.memory(latent)

        d = self.dropout(self.dec_fc(mem_latent)).view(B, 64, 7, 7)
        d = self.up1(d)
        x_hat = self.up2(d)

        return (x_hat, mem_latent) if return_latent else x_hat