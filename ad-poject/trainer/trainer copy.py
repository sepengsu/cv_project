import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ==========================
# OneEpochTrainer FP16 (train만 fp16)
# ==========================
class OneEpochTrainerFP16:
    def __init__(self, train_loader, val_loader, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler('cuda')  # FP16 scaler

    def _step(self, model, x, criterion, is_train=True):
        model.train() if is_train else model.eval()

        if hasattr(model, 'T'):  # Diffusion
            t = torch.randint(0, model.T, (x.size(0),), device=x.device)
            noise_pred, noise = model(x, t)
            loss = model.loss_function(noise_pred, noise)

        elif hasattr(model, 'discriminator'):  # GANomaly
            recon, latent, pred_fake = model(x)
            pred_real = model.discriminator(x)
            bce = nn.BCEWithLogitsLoss().to(self.device)
            d_loss_real = bce(pred_real, torch.ones_like(pred_real))
            d_loss_fake = bce(pred_fake.detach(), torch.zeros_like(pred_fake))
            d_loss = (d_loss_real + d_loss_fake) / 2
            g_adv_loss = bce(pred_fake, torch.ones_like(pred_fake))
            recon_loss = criterion(recon, x)
            loss = g_adv_loss + recon_loss + d_loss

        elif hasattr(model, 'loss_function'):  # VAE
            recon_x, mu, logvar = model(x)
            loss = model.loss_function(x, recon_x, mu, logvar)

        else:  # Plain AE
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, x)

        return loss

    def train_one_epoch(self, model, optimizer, criterion):
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for x, _ in pbar:
            x = x.to(self.device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss = self._step(model, x, criterion, is_train=True)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate_one_epoch(self, model, criterion):
        total_loss = 0
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        model.eval()
        with torch.no_grad():
            for x, _ in pbar:
                x = x.to(self.device)
                loss = self._step(model, x, criterion, is_train=False)
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)
