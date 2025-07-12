import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftedWindowBlock(nn.Module):
    def __init__(self, embed_dim, grid_size=7, window_size=3, shift_size=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)  # depthwise
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        """
        x: [B, N_patches, D] → N_patches = grid_size^2
        """
        B, N, D = x.shape
        H = W = self.grid_size
        assert N == H * W

        x = x.view(B, H, W, D)  # [B, H, W, D]

        # 1. cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # 2. local conv interaction (depthwise convolution over windows)
        x = x.permute(0, 3, 1, 2)  # [B, D, H, W]
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, D]

        # 3. reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, N, D)
        x = self.norm1(x)

        # 4. feed-forward network
        x = x + self.mlp(self.norm2(x))
        return x
