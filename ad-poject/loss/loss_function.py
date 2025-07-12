# SSIM, MS-SSIM, Charbonnier, Gradient, TV loss with autograd-safe NaN fallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ------------------------
# Gaussian Window (for SSIM)
# ------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()
    window = _2D_window.unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size).contiguous()

# ------------------------
# SSIM Loss (autograd-safe NaN fallback)
# ------------------------
def ssim_loss(img1, img2, window_size=11, eps=1e-8):
    B, C, H, W = img1.size()
    window = create_window(window_size, C).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    if torch.isnan(denominator).any():
        return torch.tensor(np.inf).to(img1.device)
    ssim_map = numerator / (denominator + eps)
    return 1 - ssim_map.mean() # Mean SSIM

# ------------------------
# MS-SSIM Loss (autograd-safe NaN fallback)
# ------------------------
def ms_ssim_loss(img1, img2, levels=3):
    weights = [0.5, 0.3, 0.2]
    loss = 0.0
    for l in range(levels):
        ssim_val = ssim_loss(img1, img2)
        if torch.isnan(ssim_val) or torch.isinf(ssim_val):
            return torch.tensor(np.inf).to(img1.device)
        loss += weights[l] * ssim_val
        img1 = F.avg_pool2d(img1, kernel_size=2)
        img2 = F.avg_pool2d(img2, kernel_size=2)
    return loss

# ------------------------
# Charbonnier Loss
# ------------------------
def charbonnier_loss(x, y, epsilon=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + epsilon ** 2))

# ------------------------
# Gradient Loss
# ------------------------
def gradient_loss(x, y):
    def gradient(img):
        dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        return dx, dy

    dx1, dy1 = gradient(x)
    dx2, dy2 = gradient(y)
    return torch.mean(torch.abs(dx1 - dx2)) + torch.mean(torch.abs(dy1 - dy2))

# ------------------------
# Total Variation (TV) Loss
# ------------------------
def tv_loss(x):
    loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return loss

# ------------------------
# laplacian edge loss
# ------------------------
def laplacian_edge_loss(x, y):
    """
    Laplacian edge-based loss (no reduction).
    Returns: (B, C, H, W) loss tensor → reduction은 외부에서 수행
    """
    B, C, H, W = x.shape

    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    x_edge = F.conv2d(x, kernel, padding=1, groups=C)
    y_edge = F.conv2d(y, kernel, padding=1, groups=C)

    loss = torch.abs(x_edge - y_edge)  # (B, C, H, W)
    return loss
