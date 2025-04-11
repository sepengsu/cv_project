from .loss_funtion import gradient_loss, tv_loss
from .loss_funtion import ssim_loss, ms_ssim_loss, charbonnier_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
# ------------------------
# Flexible Loss (조합형)
# ------------------------
class FlexibleLoss(nn.Module):
    def __init__(self, mode='mse', alpha=0.5, beta=1.0, gamma=0.1, delta=0.1, reduction='mean'):
        super(FlexibleLoss, self).__init__()
        """
        mode: 손실 조합 지정 (예: 'mse+gradient')
        reduction: 'mean' | 'sum' | 'none'
        """
        self.mode = mode.split('+')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.reduction = reduction

    def forward(self, x, y):
        loss = 0.0
        for m in self.mode:
            if m == 'mse':
                component = F.mse_loss(x, y, reduction=self.reduction)
                loss += self.beta * component

            elif m == 'l1':
                component = F.l1_loss(x, y, reduction=self.reduction)
                loss += self.beta * component

            elif m == 'charbonnier':
                diff = torch.sqrt((x - y) ** 2 + 1e-6)
                if self.reduction == 'none':
                    component = diff
                elif self.reduction == 'sum':
                    component = diff.sum()
                else:
                    component = diff.mean()
                loss += self.beta * component

            elif m == 'ssim':
                component = ssim_loss(x, y)
                if self.reduction == 'none':
                    component = torch.full((x.size(0),), component, device=x.device)
                loss += self.alpha * component

            elif m == 'ms-ssim':
                component = ms_ssim_loss(x, y)
                if self.reduction == 'none':
                    component = torch.full((x.size(0),), component, device=x.device)
                loss += self.alpha * component

            elif m == 'gradient':
                component = gradient_loss(x, y)
                if self.reduction == 'none':
                    component = torch.full((x.size(0),), component, device=x.device)
                loss += self.gamma * component

            elif m == 'tv':
                component = tv_loss(x)
                if self.reduction == 'none':
                    component = torch.full((x.size(0),), component, device=x.device)
                loss += self.delta * component

            else:
                raise NotImplementedError(f"Unknown loss mode: {m}")
        return loss
# ------------------------
# Loss Function for Diffusion Models
class FlexibleDiffusionLoss(nn.Module):
    def __init__(self, mode='mse', alpha=0.5, beta=1.0, gamma=0.1, delta=0.1, epsilon=1e-3):
        super(FlexibleDiffusionLoss, self).__init__()
        """
        mode options:
            mse, l1, charbonnier, gradient, tv
            또는 조합:
            mse+l1, mse+charbonnier, charbonnier+gradient+tv 등
        """
        self.mode = mode.split('+')
        self.alpha = alpha  # for gradient loss
        self.beta = beta    # for mse, l1, charbonnier
        self.gamma = gamma  # for tv
        self.delta = delta  # reserved (if needed)
        self.epsilon = epsilon  # for charbonnier

    def forward(self, noise_pred, noise_true):
        loss = 0.0
        for m in self.mode:
            if m == 'mse':
                loss += self.beta * F.mse_loss(noise_pred, noise_true)
            elif m == 'l1':
                loss += self.beta * F.l1_loss(noise_pred, noise_true)
            elif m == 'charbonnier':
                loss += self.beta * torch.mean(torch.sqrt((noise_pred - noise_true) ** 2 + self.epsilon ** 2))
            elif m == 'gradient':
                loss += self.alpha * gradient_loss(noise_pred, noise_true)
            elif m == 'tv':
                loss += self.gamma * tv_loss(noise_pred)
            else:
                raise NotImplementedError(f"Unknown loss type: {m}")
        return loss
# ------------------------
# Threshold 자동 계산 함수
# ------------------------
def auto_threshold(errors, ratio=0.95):
    sorted_err = torch.sort(errors)[0]
    idx = int(len(sorted_err) * ratio)
    return sorted_err[idx].item()
