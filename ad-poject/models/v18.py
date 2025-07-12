import torch
import torch.nn as nn
import torch.nn.functional as F
class KeyValueMemory(nn.Module):
    def __init__(self, mem_size=100, key_dim=128, value_dim=128, topk=10):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(mem_size, key_dim))    # 키: 검색을 위해
        self.values = nn.Parameter(torch.randn(mem_size, value_dim)) # 비용을 위해
        self.topk = topk

    def forward(self, query):  # query: (B, key_dim)
        q_norm = F.normalize(query, dim=1)
        k_norm = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q_norm, k_norm.T)  # (B, M)

        topk_sim, topk_idx = torch.topk(sim, self.topk, dim=1)       # (B, k)
        weights = F.softmax(topk_sim, dim=1)                         # (B, k)
        selected_values = self.values[topk_idx]                      # (B, k, D)

        output = torch.sum(weights.unsqueeze(-1) * selected_values, dim=1)  # (B, D)
        return output


class CenterEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        feat = self.encoder(x).view(x.size(0), -1)
        return self.fc(feat)

class StructureEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc_center = nn.Linear(latent_dim, 64 * 7 * 7)
        self.fc_skip = nn.Linear(latent_dim, 12 * 7 * 7)  # skip channel을 12로 증가
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64 + 12, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1)
        )

    def forward(self, z_center, z_skip):
        d_center = self.fc_center(z_center).view(z_center.size(0), 64, 7, 7)
        d_skip = self.fc_skip(z_skip).view(z_skip.size(0), 12, 7, 7)
        d = torch.cat([d_center, d_skip], dim=1)
        return self.decoder(d)

class CenterMemMatchingAEv3(nn.Module):
    def __init__(self, latent_dim=128, mem_size=800):
        super().__init__()
        self.center_encoder = CenterEncoder(latent_dim=latent_dim)
        self.memory_center = KeyValueMemory(mem_size=mem_size, key_dim=latent_dim, value_dim=latent_dim, topk=10)
        self.memory_skip = KeyValueMemory(mem_size=mem_size, key_dim=latent_dim, value_dim=latent_dim, topk=10)
        self.structure_encoder = CenterEncoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def crop_center(self, x, size=16):
        _, _, H, W = x.shape
        start = (H - size) // 2
        return x[:, :, start:start+size, start:start+size]

    def forward(self, x, return_latent=False):
        x_center = self.crop_center(x, 16)
        z_center = self.center_encoder(x_center)
        z_match_center = self.memory_center(z_center)

        z_skip = self.structure_encoder(x)
        z_match_skip = self.memory_skip(z_skip)

        x_hat = self.decoder(z_match_center, z_match_skip)

        if x_hat.shape[-1] != 28:
            x_hat = F.interpolate(x_hat, size=(28, 28), mode='bilinear', align_corners=False)

        return (x_hat, z_match_center) if return_latent else x_hat

@torch.no_grad()
def init_key_value_memory(model, dataloader, device, top_n=800):
    """
    CenterMemMatchingAEv3 모델용 KeyValueMemory 초기화 함수 (중앙/구조 각각 따로 초기화).
    """
    model.eval()
    center_keys, center_values = [], []
    skip_keys, skip_values = [], []

    for x, _ in dataloader:
        x = x.to(device)
        x_center = model.crop_center(x, size=16)  # ✅ 수정됨

        # 중앙 영역 feature 추출
        z_center = model.center_encoder(x_center)
        center_keys.append(z_center.cpu())
        center_values.append(z_center.cpu())

        # 구조 전체 영역 feature 추출
        z_skip = model.structure_encoder(x)
        skip_keys.append(z_skip.cpu())
        skip_values.append(z_skip.cpu())

        if len(center_keys) * x.size(0) >= top_n * 2:
            break

    # memory 크기에 맞게 자르기
    center_keys_tensor = torch.cat(center_keys, dim=0)[:top_n]
    center_values_tensor = torch.cat(center_values, dim=0)[:top_n]
    skip_keys_tensor = torch.cat(skip_keys, dim=0)[:top_n]
    skip_values_tensor = torch.cat(skip_values, dim=0)[:top_n]

    model.memory_center.keys.data = center_keys_tensor.to(device).clone()
    model.memory_center.values.data = center_values_tensor.to(device).clone()
    model.memory_skip.keys.data = skip_keys_tensor.to(device).clone()
    model.memory_skip.values.data = skip_values_tensor.to(device).clone()

    print(f"[✅] Initialized memory_center and memory_skip with {top_n} key-value pairs each.")
    return model

