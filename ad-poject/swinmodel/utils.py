import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def init_memory(model, dataloader: DataLoader, device='cuda', top_n=2000, img_size=32):
    """
    ShiftedWindowPatchAE 전용 memory 초기화 함수
    - model: ShiftedWindowPatchAE
    - dataloader: train_loader (Pullover만 포함)
    - device: cuda or cpu
    - top_n: 수집할 총 패치 수 제한
    - img_size: 32이면 padding 적용 / 28이면 padding 없이 그대로
    """
    model.eval()
    grid_size = model.grid_size
    embed_dim = model.embed_dim

    # 위치별 feature 저장 딕셔너리
    memory_features = {
        f"{i}_{j}": [] for i in range(grid_size) for j in range(grid_size)
    }

    collected = 0
    for x, _ in tqdm(dataloader, desc="Initializing ShiftedWindow Memory"):
        x = x.to(device)

        # ✅ img_size가 32이면 padding 적용 (28x28 -> 32x32)
        if img_size == 32:
            if x.shape[-1] == 28 and x.shape[-2] == 28:
                x = F.pad(x, (2, 2, 2, 2), mode='constant', value=0)
        # ✅ img_size가 28이면 padding 없이 그대로
        elif img_size == 28:
            pass  # 아무것도 안 함
        else:
            raise ValueError(f"[❌] init_memory: img_size must be 28 or 32, but got {img_size}")

        # 1. Patch embedding + position embedding
        x_patch = model.patch_embed(x)         # [B, N_patches, D]
        x_patch = model.pos_embed(x_patch)     # [B, N_patches, D]
        
        # 2. Shifted window encoding
        x_patch = model.encoder_block(x_patch) # [B, N_patches, D]
        
        # 3. 위치별 feature 분리
        B, N, D = x_patch.shape
        x_patch = x_patch.view(B, grid_size, grid_size, D)

        for i in range(grid_size):
            for j in range(grid_size):
                features = x_patch[:, i, j, :].detach().cpu()  # [B, D]
                memory_features[f"{i}_{j}"].append(features)

        collected += x.size(0)
        if collected >= top_n:
            break

    # 4. memory bank 초기화
    for key, feat_list in memory_features.items():
        feats = torch.cat(feat_list, dim=0)  # [N_total, D]
        model.memory[key].init_memory(feats)

    print("[✅] ShiftedWindowPatchAE memory initialization complete.")
