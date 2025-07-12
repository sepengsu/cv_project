import torch
import torch.nn as nn

# ✅ SEBlock (Swin attention 강화용)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weight = self.se(x)
        return x * weight

# ✅ Swin Encoder
class SwinEncoder(nn.Module):
    def __init__(self, swin_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, swin_dim // 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(swin_dim // 2, swin_dim, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(swin_dim, swin_dim, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(swin_dim, swin_dim, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            SEBlock(swin_dim)
        )

    def forward(self, x):
        return self.model(x)

class CNNEncoder(nn.Module):
    def __init__(self, cnn_dim=32):
        super().__init__()
        self.cnn_dim = cnn_dim

        self.model = nn.Sequential(
            nn.Conv2d(1, cnn_dim // 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(cnn_dim // 2, cnn_dim // 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(cnn_dim // 2, cnn_dim, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2),
        )

        # ✅ 강한 Bottleneck (cnn_dim을 기준으로 상대적 계산)
        mid_channels_1 = cnn_dim // 2
        mid_channels_2 = cnn_dim // 4
        mid_channels_3 = cnn_dim // 8

        self.bottleneck = nn.Sequential(
            nn.Conv2d(cnn_dim, mid_channels_1, kernel_size=1), nn.ReLU(),   # cnn_dim → //2
            nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=1), nn.ReLU(), # //2 → //4
            nn.Conv2d(mid_channels_2, mid_channels_3, kernel_size=1), nn.ReLU(), # //4 → //8
            nn.Dropout2d(p=0.1),
            nn.Conv2d(mid_channels_3, mid_channels_2, kernel_size=1), nn.ReLU(), # //8 → //4
            nn.Conv2d(mid_channels_2, mid_channels_1, kernel_size=1), nn.ReLU(), # //4 → //2
            nn.Conv2d(mid_channels_1, cnn_dim, kernel_size=1), nn.ReLU(),    # //2 → cnn_dim
            nn.Conv2d(cnn_dim, mid_channels_1, kernel_size=1), nn.ReLU(),    # cnn_dim → //2
            nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=1), nn.ReLU(), # //2 → //4
            nn.Conv2d(mid_channels_2, mid_channels_3, kernel_size=1), nn.ReLU(),  # //4 → //8
            nn.Dropout2d(p=0.1),
        )

        # ✅ 강한 Dropout 추가
        self.dropout = nn.Dropout2d(p=0.6, inplace=False)

    def forward(self, x):
        feat_in = self.model(x)          # (B, cnn_dim, 28, 28)
        feat_out = self.bottleneck(feat_in)
        feat_out = self.dropout(feat_out)

        # ✅ Channel mismatch 보정
        if feat_in.shape[1] != feat_out.shape[1]:
            feat_in = nn.Conv2d(feat_in.shape[1], feat_out.shape[1], kernel_size=1, bias=False).to(feat_in.device)(feat_in)

        # ✅ Residual 기반 sharpening
        residual = torch.abs(feat_in - feat_out).clamp(0, 1)
        feat_final = (1.0 - residual) ** 8  # 4제곱 sharpening 강화

        return feat_final

# ✅ SwinCNNHybridAE
class SwinCNNHybridAE(nn.Module):
    def __init__(self, swin_dim=128, cnn_dim=32, latent_noise_std=0.25,noise_std=0.25,mask_threshold = 0.5):
        super().__init__()
        self.latent_noise_std = latent_noise_std
        self.swin_dim = swin_dim
        self.cnn_dim = cnn_dim
        self.noise_std = noise_std
        self.mask_threshold = mask_threshold
        self.swin_encoder = SwinEncoder(swin_dim=swin_dim)
        self.cnn_encoder = CNNEncoder(cnn_dim=cnn_dim)

        self.swin_upsample = nn.Upsample(size=(28, 28), mode='nearest')

        self.dropout = nn.Dropout2d(p=0.6, inplace=False)

        # ✅ Conv bottleneck: in_channels = swin_dim + cnn_dim
        self.latent_conv = nn.Sequential(
            nn.Conv2d(swin_dim + cnn_dim//8, swin_dim, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(swin_dim, swin_dim, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(swin_dim, swin_dim, kernel_size=1),
            nn.LeakyReLU(0.2),
        )

        # ✅ Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(swin_dim, swin_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, swin_dim // 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(swin_dim // 2, swin_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, swin_dim // 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(swin_dim // 4, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        swin_feat = self.swin_encoder(x)          # (B, swin_dim, 7,7)
        swin_feat = self.swin_upsample(swin_feat)  # (B, swin_dim, 28,28)

        cnn_feat = self.cnn_encoder(x)             # (B, cnn_dim, 28,28)
        cnn_feat = self.dropout(cnn_feat)

        # 🔥 Mask 생성
        swin_mask = swin_feat.mean(dim=1, keepdim=True)  # (B, 1, 28, 28)
        cnn_mask = cnn_feat.mean(dim=1, keepdim=True)    # (B, 1, 28, 28)

        # 🔥 Mask 차이 계산
        mask_diff = torch.abs(swin_mask - cnn_mask)

        # 🔥 Selective Noise Injection
        noise = torch.randn_like(cnn_feat) * self.noise_std
        inject_mask = (mask_diff > self.mask_threshold).float() # mask에 따라 noise 추가
        cnn_feat = cnn_feat + noise * inject_mask  # selective noise 추가

        # 🔥 concat 후 latent bottleneck
        feat_concat = torch.cat([swin_feat, cnn_feat], dim=1)
        latent = self.latent_conv(feat_concat)

        # 🔥 latent noise 추가
        if self.training and self.latent_noise_std > 0:
            noise_latent = torch.randn_like(latent) * self.latent_noise_std
            latent = latent + noise_latent

        x_hat = self.decoder(latent)
        return x_hat