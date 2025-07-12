import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
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

class GlobalEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
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

class MemoryModule(nn.Module):
    def __init__(self, latent_dim=128, memory_size=100, topk=10):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, latent_dim))
        self.topk = topk

    def forward(self, x):
        mem_norm = F.normalize(self.memory, dim=1)
        x_norm = F.normalize(x, dim=1)
        sim = torch.matmul(x_norm, mem_norm.T)
        topk_sim, topk_idx = torch.topk(sim, self.topk, dim=1)
        weights = F.softmax(topk_sim, dim=1)
        selected = self.memory[topk_idx]
        return torch.sum(weights.unsqueeze(-1) * selected, dim=1)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 7→14
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.ReLU(),   # 14→28
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, z):
        d = self.fc(z).view(z.size(0), 64, 7, 7)
        return self.decoder(d)

class DualEncoderMatchingAE(nn.Module):
    def __init__(self, latent_dim_each=64, memory_size=100):
        super().__init__()
        self.center_encoder = CenterEncoder(latent_dim=latent_dim_each)
        self.global_encoder = GlobalEncoder(latent_dim=latent_dim_each)
        self.memory = MemoryModule(latent_dim=latent_dim_each * 2, memory_size=memory_size)
        self.decoder = Decoder(latent_dim=latent_dim_each * 2)

    def crop_center(self, x, size=14):
        _, _, H, W = x.shape
        start = (H - size) // 2
        return x[:, :, start:start+size, start:start+size]

    def forward(self, x, return_latent=False):
        x_center = self.crop_center(x, 14)                  # (B, 1, 14, 14)
        z_center = self.center_encoder(x_center)            # (B, 64)
        z_global = self.global_encoder(x)                   # (B, 64)
        z_combined = torch.cat([z_center, z_global], dim=1) # (B, 128)

        z_match = self.memory(z_combined)                   # (B, 128)
        x_hat = self.decoder(z_match)                       # (B, 1, 28, 28)

        return (x_hat, z_match) if return_latent else x_hat


import torch
from sklearn.cluster import KMeans

def pretrain_memory(model, data_loader, device="cuda"):
    model.eval()  # inference 모드
    latent_list = []

    # latent vector들을 모두 수집
    with torch.no_grad():
        for batch in data_loader:
            x, _ = batch  # 라벨이 있을 경우 무시
            x = x.to(device)
            # 중앙 crop
            x_center = model.crop_center(x, 14)
            # 각각 latent vector 계산
            z_center = model.center_encoder(x_center)  # (B, latent_dim_each)
            z_global = model.global_encoder(x)           # (B, latent_dim_each)
            # Concatenate → (B, latent_dim_each * 2)
            z_combined = torch.cat([z_center, z_global], dim=1)
            latent_list.append(z_combined.cpu())

    latent_all = torch.cat(latent_list, dim=0)  # shape: [N, latent_dim_total]
    latent_all_np = latent_all.numpy()

    memory_size = model.memory.memory.shape[0]
    # K-means clustering
    kmeans = KMeans(n_clusters=memory_size, random_state=42).fit(latent_all_np)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)

    # MemoryModule에 결과 복사 (모델이 GPU에 있다면 GPU로 옮기기)
    model.memory.memory.data.copy_(centers.to(device))
    model.train()  # 모델을 다시 train mode로 변경

    print("Memory pre-initialization complete.")
