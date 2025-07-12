import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_function import ssim_loss, ms_ssim_loss, gradient_loss, tv_loss, laplacian_edge_loss
from .center import center_crop_loss, center_weighted_loss, create_center_weight_map
from .fft import fft_loss

class FlexibleLoss(nn.Module):
    def __init__(self, mode='mse', loss_weights=None, reduction='mean', epoch_threshold=10, crop_size=16, sigma=5):
        super().__init__()
        self.modes = mode.lower().split('+')
        self.reduction = reduction
        self.base_weights = loss_weights if loss_weights else {m: 1.0 for m in self.modes}
        self.epoch_threshold = epoch_threshold
        self.crop_size = crop_size
        self.sigma = sigma
        self.map = create_center_weight_map(28, sigma).to('cuda' if torch.cuda.is_available() else 'cpu')

    def reduce(self, x):
        if x is None:
            return None
        if not torch.isfinite(x).all():
            return None
        if self.reduction == 'none':
            # 🔥 여기서 (B,) 형태로 정리
            return x.view(x.size(0), -1).mean(dim=1) if x.dim() > 1 else x.view(-1)
        elif self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()

    def forward(self, x, y,epoch=100):
        valid_losses = {}
        active_weights = {}

        if epoch< self.epoch_threshold:
            raw = F.mse_loss(x, y, reduction='none')
            loss = self.reduce(raw)
            return loss

        for m in self.modes:
            w = self.base_weights.get(m, 1.0)
            if w == 0:
                continue
            try:
                if m == 'mse':
                    raw = F.mse_loss(x, y, reduction='none')
                    loss = self.reduce(raw)
                elif m == 'charbonnier':
                    raw = torch.sqrt((x - y) ** 2 + 1e-6)
                    loss = self.reduce(raw)
                elif m == 'ssim':
                    raw = ssim_loss(x, y)
                    loss = self.reduce(raw)
                elif m == 'ms-ssim':
                    raw = ms_ssim_loss(x, y)
                    loss = self.reduce(raw)
                elif m == 'gradient':
                    raw = gradient_loss(x, y)
                    loss = self.reduce(raw)
                elif m == 'tv':
                    raw = tv_loss(x)
                    loss = self.reduce(raw)
                elif m == 'edge':
                    raw = laplacian_edge_loss(x, y)
                    loss = self.reduce(raw)
                elif m == 'center_crop':
                    raw = center_crop_loss(x, y, size=self.crop_size)
                    loss = self.reduce(raw)
                elif m == 'center_weighted':
                    raw = center_weighted_loss(x, y,weight_map=self.map)
                    loss = self.reduce(raw)
                elif m == 'fft':
                    raw = fft_loss(x, y)
                    loss = self.reduce(raw)
                elif m == 'noise':
                    continue
                else:
                    raise ValueError(f"[FlexibleLoss] Unknown loss type: {m}")

                if loss is not None and torch.isfinite(loss).all():
                    valid_losses[m] = loss
                    active_weights[m] = w

            except Exception as e:
                print(f"[❌ Error] {m} loss computation failed: {e}")

        total_weight = sum(active_weights.values())

        if total_weight == 0:
            print("[‼️ Fallback] All loss terms invalid or zero-weighted → dummy loss used")
            dummy = F.mse_loss(x, y, reduction='none').view(x.size(0), -1).mean(dim=1)
            return dummy * 0.0 + 1.0 if self.reduction == 'none' else dummy.mean() * 0.0 + 1.0

        total_loss = None
        for m, loss_val in valid_losses.items():
            scaled_w = active_weights[m] / total_weight
            weighted_loss = scaled_w * loss_val
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        return total_loss
