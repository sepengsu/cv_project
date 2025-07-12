import torch
import torch.nn as nn
import torch.nn.functional as F

class KeyValueMemory(nn.Module):
    def __init__(self, mem_size=100, key_dim=64, value_dim=64, topk=5):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(mem_size, key_dim))
        self.values = nn.Parameter(torch.randn(mem_size, value_dim))
        self.topk = topk

    def forward(self, x):
        B, C, H, W = x.shape
        query = x.view(B, C, -1).mean(dim=2)
        q_norm = F.normalize(query, dim=1)
        k_norm = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q_norm, k_norm.T)
        topk_sim, topk_idx = torch.topk(sim, self.topk, dim=1)
        weights = F.softmax(topk_sim, dim=1)
        selected = self.values[topk_idx]  # (B, k, C)
        matched = torch.sum(weights.unsqueeze(-1) * selected, dim=1)  # (B, C)
        return matched.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        x_down = self.pool(x)
        return x, x_down

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class UNetAutoEncoderWithMem(nn.Module):
    def __init__(self, mem_size=400, base_dim=32):
        super().__init__()
        self.enc1 = EncoderBlock(1, base_dim)              # 28 → 14
        self.enc2 = EncoderBlock(base_dim, base_dim * 2)   # 14 → 7

        self.mem1 = KeyValueMemory(mem_size, base_dim, base_dim)
        self.mem2 = KeyValueMemory(mem_size, base_dim * 2, base_dim * 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(base_dim * 4, base_dim * 4, kernel_size=3, padding=1), nn.ReLU()
        )

        self.dec2 = DecoderBlock(base_dim * 4, base_dim * 2, base_dim * 2)  # 7 → 14
        self.dec1 = DecoderBlock(base_dim * 2, base_dim, base_dim)          # 14 → 28

        self.output_layer = nn.Conv2d(base_dim, 1, kernel_size=1)

    def forward(self, x):
        s1, x = self.enc1(x)  # 28 -> 14
        s2, x = self.enc2(x)  # 14 -> 7

        m2 = self.mem2(s2)
        x = self.bottleneck(x)  # 7x7
        x = self.dec2(x, m2)    # match skip2 with mem

        m1 = self.mem1(s1)
        x = self.dec1(x, m1)    # match skip1 with mem

        return self.output_layer(x)

@torch.no_grad()
def init_multi_feature_memory(model, dataloader, device, top_n=400):
    """
    UNetAutoEncoderWithMem 전용 KeyValueMemory 초기화 함수
    mem2 (encoder skip) + mem_bottleneck (bottleneck feature)에 대해 초기화 수행
    """
    model.eval()
    mem2_keys, bottleneck_keys = [], []

    for x, _ in dataloader:
        x = x.to(device)
        with torch.no_grad():
            _, x = model.enc1(x)     # 28 → 14
            s2, x = model.enc2(x)    # 14 → 7
            bottleneck_feat = model.bottleneck(x)

        mem2_keys.append(s2.view(s2.size(0), s2.size(1), -1).mean(dim=2).cpu())
        bottleneck_keys.append(bottleneck_feat.view(bottleneck_feat.size(0), bottleneck_feat.size(1), -1).mean(dim=2).cpu())

        if len(mem2_keys) * x.size(0) >= top_n * 2:
            break

    mem2_tensor = torch.cat(mem2_keys, dim=0)[:top_n].to(device)
    bottleneck_tensor = torch.cat(bottleneck_keys, dim=0)[:top_n].to(device)

    model.mem2.keys.data = mem2_tensor.clone()
    model.mem2.values.data = mem2_tensor.clone()
    model.mem_bottleneck.keys.data = bottleneck_tensor.clone()
    model.mem_bottleneck.values.data = bottleneck_tensor.clone()

    print(f"[✅] Initialized KeyValueMemory for mem2 and mem_bottleneck with {top_n} key-value pairs each.")
    return model

