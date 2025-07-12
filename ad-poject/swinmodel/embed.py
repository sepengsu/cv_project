import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # e.g., 28//4=7
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.proj(x)  # [B, embed_dim, H', W'] → H'=W'=grid_size
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, D]
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim, learnable=True):
        super().__init__()
        if learnable:
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        else:
            self.register_buffer("pos_embed", self._build_sinusoidal(num_patches, embed_dim))

    def forward(self, x):
        # x: [B, N_patches, D]
        return x + self.pos_embed

    def _build_sinusoidal(self, num_patches, dim):
        position = torch.arange(num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(1, num_patches, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
