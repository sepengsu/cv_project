import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class UNetMemAEv9Structured(nn.Module):
    def __init__(self, memory_size=400, memory_dim=256):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # 🔵 Structure Encoder
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 🔶 Texture Encoder (FFT)
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.to_latent = nn.Linear(512, memory_dim)  # 256(struct) + 256(text)

        # 🧠 Memory
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))

        # 🟡 Decoder (memory → 28×28)
        self.up1 = nn.ConvTranspose2d(memory_dim, 128, 4, stride=2, padding=1)
        self.dec1 = self.conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = self.conv_block(32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def fft_magnitude(self, x):
        fft = torch.fft.fft2(x)
        real = fft.real
        imag = fft.imag
        out = torch.cat([real, imag], dim=1)  # concat along channel dim: [B, 2, H, W]
        return out

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
        )

    def forward(self, x, return_latent=False):
        B = x.size(0)

        # 🔵 Structure branch
        struct_feat = self.structure_encoder(x).view(B, -1)

        # 🔶 Texture branch (FFT)
        fft = self.fft_magnitude(x)
        text_feat = self.texture_encoder(fft).view(B, -1)

        # Concatenate → latent
        f_latent = self.to_latent(torch.cat([struct_feat, text_feat], dim=1))  # (B, memory_dim)

        # 🧠 Memory attention
        attn = F.softmax(torch.matmul(f_latent, self.memory.T), dim=1)
        f_mem = torch.matmul(attn, self.memory)

        # 🟡 Decoder
        d = f_mem.view(B, self.memory_dim, 1, 1).expand(-1, self.memory_dim, 7, 7)
        d = self.up1(d)
        d = self.dec1(d)
        d = self.up2(d)
        d = self.dec2(d)

        x_hat = self.final(d)  # (B, 1, 28, 28)

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

        # 🔵 Structure latent
        struct_feat = model.structure_encoder(x).view(x.size(0), -1)

        # 🔶 Texture latent (FFT 기반)
        fft = model.fft_magnitude(x)
        text_feat = model.texture_encoder(fft).view(x.size(0), -1)

        # Concatenate → latent
        f_latent = model.to_latent(torch.cat([struct_feat, text_feat], dim=1))  # (B, memory_dim)
        latent_list.append(f_latent.cpu())

    # 🔄 KMeans 초기화
    all_latents = torch.cat(latent_list, dim=0).numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(all_latents)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    # (선택) 정규화
    centroids = F.normalize(centroids, dim=1)

    model.memory.data.copy_(centroids)
    print(f"✅ Memory initialized (KMeans-{num_clusters}) from class 2 | dim={model.memory_dim}")
