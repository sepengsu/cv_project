# transformer.py (Vision Transformer 기반 Reconstruction)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, emb_dim=256):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(1, emb_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, emb_dim]
        return x

class TransformerAnomalyDetector(nn.Module):
    def __init__(self, img_size=28, patch_size=4, emb_dim=256, depth=4, n_heads=4):
        super(TransformerAnomalyDetector, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, patch_size * patch_size),
            nn.ReLU()
        )
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        B = x.size(0)
        patches = self.patch_embed(x)  # [B, N, emb_dim]
        features = self.transformer(patches)  # [B, N, emb_dim]
        recon_patches = self.decoder(features)  # [B, N, patch_size*patch_size]

        # 재구성된 패치를 원본 이미지로 복원
        recon_patches = recon_patches.view(B, -1, self.patch_size, self.patch_size)
        recon_img = recon_patches.permute(0, 1, 2, 3).contiguous()
        n_patches_per_row = self.img_size // self.patch_size
        recon_img = recon_img.view(B, 1, n_patches_per_row * self.patch_size, n_patches_per_row * self.patch_size)
        return recon_img
