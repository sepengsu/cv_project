# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ------------------------
# Gaussian Window
# ------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        torch.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()
    window = _2D_window.unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size).contiguous()

# ------------------------
# SSIM Loss
# ------------------------
def ssim_loss(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + 0.01**2) * (2 * sigma12 + 0.03**2)) / ((mu1_sq + mu2_sq + 0.01**2) * (sigma1_sq + sigma2_sq + 0.03**2))
    return 1 - ssim_map.mean()

# ------------------------
# MS-SSIM (Multi-scale SSIM)
# ------------------------
def ms_ssim_loss(img1, img2, levels=3):
    weights = [0.5, 0.3, 0.2]  # level별 가중치
    loss = 0.0
    for l in range(levels):
        loss += weights[l] * ssim_loss(img1, img2)
        img1 = F.avg_pool2d(img1, kernel_size=2)
        img2 = F.avg_pool2d(img2, kernel_size=2)
    return loss

# ------------------------
# Perceptual Loss
# ------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        # vgg = models.vgg16(pretrained=True).features
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_layers = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.resize = resize

    def forward(self, x, y):
        if self.resize:
            x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224,224), mode='bilinear', align_corners=False)
        return F.l1_loss(self.vgg_layers(x), self.vgg_layers(y))

# ------------------------
# Flexible Mixed Loss Class
# ------------------------
class FlexibleLoss(nn.Module):
    def __init__(self, mode='mse', alpha=0.5, beta=1.0, gamma=0.1):
        super(FlexibleLoss, self).__init__()
        assert mode in ['mse', 'ssim', 'perceptual', 
                'mse+ssim', 'mse+perceptual', 'ssim+perceptual', 'mse+ssim+perceptual']

        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, x, y):
        loss = 0

        if 'mse' in self.mode:
            rec_loss = F.mse_loss(x, y)
            loss += self.beta * rec_loss

        if 'ssim' in self.mode:
            ssim = ssim_loss(x, y)
            loss += self.alpha * ssim

        if 'perceptual' in self.mode:
            perceptual = self.perceptual_loss(x, y)
            loss += self.gamma * perceptual

        return loss
# ------------------------
# Auto Threshold Estimator
# ------------------------
def auto_threshold(errors, ratio=0.95):
    """
    errors: list or tensor of reconstruction errors
    ratio: 몇 %를 정상으로 볼지 (ex: 0.95 -> 상위 5%는 이상)
    """
    sorted_err = torch.sort(errors)[0]
    idx = int(len(sorted_err) * ratio)
    return sorted_err[idx].item()
