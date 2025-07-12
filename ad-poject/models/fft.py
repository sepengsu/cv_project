import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ FFT Encoder
class FFTEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_channels, 3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_mag = torch.abs(torch.fft.fftshift(x_fft))
        if x_mag.dim() == 3:
            x_mag = x_mag.unsqueeze(1)
        return self.encoder(x_mag)


# ✅ FFT Memory
class FFTMemory(nn.Module):
    def __init__(self, mem_size=400, dim=128, topk=5):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(mem_size, dim), requires_grad=False)
        self.values = nn.Parameter(torch.randn(mem_size, dim), requires_grad=False)
        self.topk = topk

    def forward(self, q):
        if q.dim() == 4:
            q = q.mean(dim=[2, 3])
        q = F.normalize(q, dim=1)
        k = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q, k.T)
        top_val, top_idx = torch.topk(sim, self.topk, dim=1)
        weights = F.softmax(top_val, dim=1)
        v_selected = self.values[top_idx]
        weighted = (v_selected * weights.unsqueeze(-1)).sum(dim=1)
        return weighted.unsqueeze(-1).unsqueeze(-1)


# ✅ FFT Decoder
class FFTDecoder(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)


# ✅ FFT Branch AutoEncoder (with full integration)
class FFTMemAutoEncoderBranch(nn.Module):
    def __init__(self, latent_dim=128, mem_size=400):
        super().__init__()
        self.encoder = FFTEncoder(in_channels=1, out_channels=latent_dim)
        self.memory = FFTMemory(mem_size=mem_size, dim=latent_dim, topk=5)
        self.decoder = FFTDecoder(in_channels=latent_dim)

    def forward(self, x):
        feat = self.encoder(x)
        mem = self.memory(feat)
        recon = self.decoder(mem)
        return recon


# ✅ Unified Model: Main + FFT Branch
class UnifiedMemAutoEncoder(nn.Module):
    def __init__(self, main_branch, fft_branch):
        super().__init__()
        self.main_branch = main_branch
        self.fft_branch = fft_branch

    def forward(self, x, return_all=False):
        x_hat_main = self.main_branch(x)
        x_hat_fft = self.fft_branch(x)
        if return_all:
            return x_hat_main, x_hat_fft
        return x_hat_main


