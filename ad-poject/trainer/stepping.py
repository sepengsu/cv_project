import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseStep:
    def __init__(self, device):
        self.device = device

    def forward(self, model, x, criterion=None, is_train=False, optimizer_D=None):
        raise NotImplementedError("Subclass must implement this")


class AENormalStep(BaseStep):
    def forward(self, model, x, criterion=None, is_train=False, optimizer_D=None):
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        return criterion(output, x)


class DiffusionStep(BaseStep):
    def forward(self, model, x, criterion=None, is_train=False, optimizer_D=None):
        t = torch.randint(0, model.T, (x.size(0),), device=self.device)
        noise_pred, noise = model(x, t)
        return model.loss_function(noise_pred, noise)


class GANomalyStep(BaseStep):
    def __init__(self, device, scaler):
        super().__init__(device)
        self.scaler = scaler
        self.bce = nn.BCEWithLogitsLoss().to(device)

    def forward(self, model, x, criterion=None, is_train=False, optimizer_D=None):
        z = model.encoder(x)
        x_hat = model.decoder(z)
        z_hat = model.encoder(x_hat.detach())

        # Discriminator step
        d_real, _ = model.discriminator(x)
        d_fake, _ = model.discriminator(x_hat.detach())

        d_loss_real = self.bce(d_real, torch.ones_like(d_real))
        d_loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_loss_real + d_loss_fake) / 2

        if is_train:
            optimizer_D.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.scaler.step(optimizer_D)

        # Generator step
        d_fake_for_G, _ = model.discriminator(x_hat)
        recon_loss = criterion(x_hat, x)
        latent_loss = F.l1_loss(z_hat, z)
        adv_loss = self.bce(d_fake_for_G, torch.ones_like(d_fake_for_G))

        return recon_loss + latent_loss + adv_loss


class StepFactory:
    @staticmethod
    def create(model, device, scaler=None):
        if hasattr(model, 'T'):
            return DiffusionStep(device)
        elif hasattr(model, 'discriminator'):
            return GANomalyStep(device, scaler)
        else:
            return AENormalStep(device)
