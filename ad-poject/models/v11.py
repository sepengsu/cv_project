import torch
import torch.nn as nn
import torch.nn.functional as F
class UNetMemAEv11CentralGated(nn.Module):
    def __init__(self, memory_size=400, memory_dim=256):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # 🔵 구조 Encoder
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # → (B, 128, 1, 1)
        )

        # 🔶 중앙 질감 Encoder (직접 decoder로 전달)
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 → 7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # ✅ 추가됨 → (B, 64, 1, 1)
        )

        self.to_latent = nn.Linear(128 + 64, memory_dim)  # ✅ 입력차원 수정: 192 → 256
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))

        # 🟡 Decoder
        self.up1 = nn.ConvTranspose2d(memory_dim + 64, 128, 4, stride=2, padding=1)
        self.dec1 = self.conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = self.conv_block(32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def crop_center(self, x, size=14):
        _, _, H, W = x.shape
        start = (H - size) // 2
        return x[:, :, start:start+size, start:start+size]

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
        )

    def forward(self, x, return_latent=False):
        B = x.size(0)

        # 🔵 구조 → memory latent
        f_struct = self.structure_encoder(x).view(B, -1)  # (B, 128)

        # 🔶 중앙 질감
        x_crop = self.crop_center(x, 14)
        f_text = self.texture_encoder(x_crop).view(B, -1)  # (B, 64)

        # 🧠 latent + attention
        f_latent = self.to_latent(torch.cat([f_struct, f_text], dim=1))  # (B, 192) → (B, 256)
        attn = F.softmax(torch.matmul(f_latent, self.memory.T), dim=1)
        f_mem = torch.matmul(attn, self.memory)

        # 🟡 decoder input + texture feature concat
        d = f_mem.view(B, self.memory_dim, 1, 1).expand(-1, self.memory_dim, 7, 7)
        f_text_feat = f_text.view(B, 64, 1, 1).expand(-1, 64, 7, 7)  # ✅ 채널 맞춰 concat
        d = torch.cat([d, f_text_feat], dim=1)  # (B, 256+64, 7, 7)

        d = self.up1(d)
        d = self.dec1(d)
        d = self.up2(d)
        d = self.dec2(d)
        x_hat = self.final(d)

        return (x_hat, f_latent, f_mem) if return_latent else x_hat


from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F

@torch.no_grad()
def init(model, dataloader, device='cuda', num_clusters=400):
    model.eval()
    latent_list = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # ✅ Class 2만 사용
        mask = (y == 2)
        if mask.sum() == 0:
            continue
        x = x[mask]

        # 🔵 구조 feature
        struct_feat = model.structure_encoder(x).view(x.size(0), -1)

        # 🔶 중앙 질감 feature
        x_crop = model.crop_center(x, size=14)  # 중앙 crop
        text_feat = model.texture_encoder(x_crop).view(x.size(0), -1)

        # 🧠 통합 latent → to_latent
        f_latent = model.to_latent(torch.cat([struct_feat, text_feat], dim=1))
        latent_list.append(f_latent.cpu())

    # 🔄 모든 latent → KMeans
    all_latents = torch.cat(latent_list, dim=0).numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(all_latents)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    # (선택적) 정규화
    centroids = F.normalize(centroids, dim=1)

    model.memory.data.copy_(centroids)
    print(f"✅ Memory initialized (KMeans-{num_clusters}) from class 2 | dim={model.memory_dim}")
