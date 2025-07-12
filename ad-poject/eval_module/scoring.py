import torch
import torch.nn.functional as F

class AnomalyScoreComputer:
    def __init__(self, model, loss_fn, device):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def compute(self, x):
        model = self.model
        model.eval()
        x = x.to(self.device)

        # 1. fAnoGAN or latent-based anomaly model
        if hasattr(model, 'name') and model.name == 'fAnoGAN':
            z = model.encode(x)
            x_hat = model.generate(z)
            d_score = model.discriminate(x_hat)
            return 1- torch.sigmoid(d_score).view(-1)

        # 2. DDPM 등 noise prediction 기반
        elif hasattr(model, "T"):
            t = torch.randint(0, model.T, (x.size(0),), device=self.device)
            noise_pred, noise = model(x, t)
            score = F.mse_loss(noise_pred, noise, reduction='none')
            return score.view(score.size(0), -1).mean(dim=1)

        # 4. 일반 AE 구조
        else:
            x_hat = model(x)
            if isinstance(x_hat, tuple):
                x_hat = x_hat[0]
            score = self.loss_fn(x_hat, x)  # 이미 FlexibleLoss가 (B,) or scalar로 정리함
            return score
