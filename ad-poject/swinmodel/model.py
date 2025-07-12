# ✅ 최종 수정버전: ShiftedWindowPatchAE (32x32 패딩 + 2x2 패치 + 28x28 크롭)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory import KeyValueMemory
from .embed import PatchEmbedding, PositionEmbedding
from .decoder import PatchDecoder, ReconstructImageFromPatches, WeakenedDecoder
from .block import ShiftedWindowBlock

class ShiftedWindowPatchAE(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=2,
        embed_dim=128,
        mem_size=100,
        window_size=3,
        shift_size=1,
        memory_topk=5,
        use_memory=True,
        in_channels=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.use_memory = use_memory
        self.memory_topk = memory_topk

        # ✅ Patch Embedding + Positional Encoding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = PositionEmbedding(num_patches=self.num_patches, embed_dim=embed_dim)

        # ✅ Shifted Window Encoder Block
        self.encoder_block = ShiftedWindowBlock(embed_dim, grid_size=self.grid_size,
                                                window_size=window_size, shift_size=shift_size)

        # ✅ Position-aware Memory (patch별 memory bank)
        if self.use_memory:
            self.memory = nn.ModuleDict({
                f"{i}_{j}": KeyValueMemory(mem_size, key_dim=embed_dim, value_dim=embed_dim, topk=memory_topk)
                for i in range(self.grid_size) for j in range(self.grid_size)
            })

        # ✅ Decoder (patch-wise → full image)
        self.decoder = WeakenedDecoder(embed_dim=embed_dim, patch_size=patch_size)
        self.reconstructor = ReconstructImageFromPatches(img_size=img_size, patch_size=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        # ✅ 28x28 입력일 경우 32x32로 패딩
        if H == 28 and W == 28:
            pad = (2, 2, 2, 2)
            x = F.pad(x, pad, mode='constant', value=0)

        # Patch Embedding
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        # Shifted Window Encoder
        x = self.encoder_block(x)

        # Memory Matching (optional)
        if self.use_memory:
            B, N, D = x.shape
            grid_H = grid_W = self.grid_size
            x = x.view(B, grid_H, grid_W, D)

            matched_list = []
            for i in range(grid_H):
                for j in range(grid_W):
                    patch_feat = x[:, i, j, :]
                    key = f"{i}_{j}"
                    memory_bank = self.memory[key]
                    matched_feat = memory_bank(patch_feat)
                    matched_list.append(matched_feat)

            x = torch.stack(matched_list, dim=1)

        # Decoder
        patches = self.decoder(x)
        x_recon = self.reconstructor(patches)  # [B, 1, 32, 32]

        # ✅ 32x32 -> 28x28 center crop
        x_recon = x_recon[:, :, 2:30, 2:30]

        return x_recon

"""
📢 질문: 32x32 패딩 후 복원하고 28x28로 crop하는 게 과연 의미가 있을까?

✅ 의미가 있습니다.
- 32x32로 padding하면 모델이 더 fine-grained spatial 구조를 학습할 수 있습니다.
- 2x2 patch로 16x16 grid를 형성하면, 더 작은 anomaly (ex: 글자, 구김, 주름)까지 민감하게 복원/감지할 수 있습니다.
- 이후 center crop으로 28x28 input과 정확하게 비교 가능하므로, evaluation consistency도 유지됩니다.

결론: 성능 향상 가능성 매우 높으며, 특히 미세 anomaly 감지에서 효과가 큽니다.
"""
