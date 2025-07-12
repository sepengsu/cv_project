import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class KeyValueMemory(nn.Module):
    def __init__(self, mem_size=100, key_dim=128, value_dim=128, topk=3):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(mem_size, key_dim))
        self.values = nn.Parameter(torch.randn(mem_size, value_dim))
        self.topk = topk

    def forward(self, query):
        if query.dim() == 2:
            return self._match_vector(query)
        elif query.dim() == 4:
            return self._match_spatial(query)
        else:
            raise ValueError("Unsupported query shape")

    def _match_vector(self, query):
        q_norm = F.normalize(query, dim=1)
        k_norm = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q_norm, k_norm.T)
        topk_sim, topk_idx = torch.topk(sim, self.topk, dim=1)
        weights = F.softmax(topk_sim, dim=1)
        selected_values = self.values[topk_idx]
        output = torch.sum(weights.unsqueeze(-1) * selected_values, dim=1)
        return output

    def _match_spatial(self, query):
        B, C, H, W = query.shape
        q = query.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        q = F.normalize(q, dim=-1)
        k = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q, k.T)  # (B, H*W, M)
        top_val, top_idx = torch.topk(sim, self.topk, dim=-1)
        weights = F.softmax(top_val, dim=-1)  # (B, H*W, topk)
        v_selected = self.values[top_idx]     # (B, H*W, topk, C)
        weighted = (v_selected * weights.unsqueeze(-1)).sum(dim=2)  # (B, H*W, C)
        return weighted.permute(0, 2, 1).view(B, C, H, W)

class HardMemory(nn.Module):
    def __init__(self, mem_size=100, mem_dim=128):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, mem_dim))

    def forward(self, x):
        if x.dim() == 2:
            x_norm = F.normalize(x, dim=1)
            mem_norm = F.normalize(self.memory, dim=1)
            sim = torch.matmul(x_norm, mem_norm.T)
            max_idx = torch.argmax(sim, dim=1)
            return self.memory[max_idx]  # (B, C)
        elif x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            x_norm = F.normalize(x_flat, dim=-1)
            mem_norm = F.normalize(self.memory, dim=1)  # (M, C)
            sim = torch.matmul(x_norm, mem_norm.T)  # (B, H*W, M)
            max_idx = torch.argmax(sim, dim=-1)  # (B, H*W)
            mem_selected = self.memory[max_idx]  # (B, H*W, C)
            return mem_selected.permute(0, 2, 1).view(B, C, H, W)
        else:
            raise ValueError("Unsupported input shape for HardMemory")

class CenterEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 → 14
            nn.Conv2d(32, latent_dim, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 → 14
        )

    def forward(self, x):
        return self.encoder(x)  # (B, latent_dim, H, W)

class StructureEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 → 14
            nn.Conv2d(16, out_channels, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)   # 14 → 7
        )

    def forward(self, x):
        return self.encoder(x)  # (B, C, 7, 7)
class Decoder(nn.Module):
    def __init__(self, in_ch_center=128, in_ch_skip=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch_center + in_ch_skip, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 7 → 14
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),  # 14 → 28
            nn.Conv2d(16, 1, 3, padding=1)  # 최종 복원
        )

    def forward(self, z_center, z_skip):
        d = torch.cat([z_center, z_skip], dim=1)
        return self.decoder(d)


class CenterMemMatchingAEv4(nn.Module):
    def __init__(self, latent_dim=128, mem_size=800, crop_size=24):
        super().__init__()
        self.crop_size = crop_size
        self.center_encoder = CenterEncoder(latent_dim=latent_dim)
        self.memory_center = KeyValueMemory(mem_size=mem_size, key_dim=latent_dim, value_dim=latent_dim, topk=10)
        self.memory_skip = HardMemory(mem_size=mem_size, mem_dim=latent_dim)
        self.structure_encoder = StructureEncoder(in_channels=1, out_channels=latent_dim)
        self.decoder = Decoder(in_ch_center=latent_dim, in_ch_skip=latent_dim)

    def crop_center(self, x):
        size = self.crop_size
        _, _, H, W = x.shape
        assert size <= H and size <= W, "Crop size must be less than or equal to input size"

        start_h = (H - size) // 2
        start_w = (W - size) // 2
        x_crop = x[:, :, start_h:start_h+size, start_w:start_w+size]

        pad_h = H - size
        pad_w = W - size
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return F.pad(x_crop, (pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x, return_latent=False):
        x_center = self.crop_center(x)
        z_center = self.center_encoder(x_center)
        z_match_center = self.memory_center(z_center)

        z_skip = self.structure_encoder(x)
        z_match_skip = self.memory_skip(z_skip)
        x_hat = self.decoder(z_match_center, z_match_skip)
        return (x_hat, z_match_center) if return_latent else x_hat

@torch.no_grad()
def init_key_value_memory(model, dataloader, device, mem_size=800, top_n=2000, mode='kmeans'):
    model.eval()
    center_feats, skip_feats = [], []

    for x, _ in dataloader:
        x = x.to(device)
        x_center = model.crop_center(x)
        z_center = model.center_encoder(x_center)
        z_center_flat = z_center.mean(dim=[2, 3])
        center_feats.append(z_center_flat.cpu())

        z_skip = model.structure_encoder(x)
        z_skip_flat = z_skip.mean(dim=[2, 3])
        skip_feats.append(z_skip_flat.cpu())

        if len(center_feats) * x.size(0) >= top_n:
            break

    center_feats = torch.cat(center_feats, dim=0)
    skip_feats = torch.cat(skip_feats, dim=0)

    if mode == 'kmeans':
        print(f"[🔍] Running KMeans for {mem_size} clusters...")
        kmeans = KMeans(n_clusters=mem_size, n_init='auto', random_state=42).fit(center_feats.numpy())
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    elif mode == 'mean':
        centers = center_feats[:mem_size]
    else:
        raise ValueError("mode must be 'kmeans' or 'mean'")

    model.memory_center.keys.data = centers.to(device).clone()
    model.memory_center.values.data = centers.to(device).clone()
    model.memory_skip.memory.data = skip_feats[:mem_size].to(device).clone()
    model.memory_center.keys.requires_grad = True
    model.memory_center.values.requires_grad = True

    print(f"[✅] Initialized memory_center (key/value) and memory_skip (hard) with {mem_size} samples (mode={mode}).")
    return model

