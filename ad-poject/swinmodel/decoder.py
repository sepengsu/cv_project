import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. PatchDecoder: patch-level latent → patch pixels (e.g., 4x4)
class PatchDecoder(nn.Module):
    def __init__(self, embed_dim=128, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size),
        )

    def forward(self, x):
        # x: [B, N_patches, D]
        B, N, D = x.shape
        x = self.decoder(x)  # [B, N_patches, patch_size*patch_size]
        x = x.view(B, N, self.patch_size, self.patch_size)
        return x
    

# 2. ReconstructImageFromPatches: patches → full image
class ReconstructImageFromPatches(nn.Module):
    def __init__(self, img_size=28, patch_size=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

    def forward(self, x):
        # x: [B, N_patches, patch_H, patch_W]
        B, N, H, W = x.shape
        assert H == self.patch_size and W == self.patch_size
        x = x.view(B, self.grid_size, self.grid_size, H, W)
        x = x.permute(0, 1, 3, 2, 4)  # [B, grid_H, patch_H, grid_W, patch_W]
        x = x.contiguous().view(B, 1, self.img_size, self.img_size)
        return x
    

class WeakenedDecoder(nn.Module):
    def __init__(self, embed_dim=64, patch_size=4, grid_size=7):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size  # 28/4=7

        reduced_dim = embed_dim // 2

        self.decode_layers = nn.Sequential(
            nn.Linear(embed_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, patch_size * patch_size),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N_patches, embed_dim)
        Returns:
            patches: (B, N_patches, patch_size, patch_size)
        """
        B, N, D = x.shape
        x = self.decode_layers(x)  # (B, N_patches, patch_size*patch_size)
        x = x.view(B, N, self.patch_size, self.patch_size)  # (B, N_patches, patch_H, patch_W)
        return x


