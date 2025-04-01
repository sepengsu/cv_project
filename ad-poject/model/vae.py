# vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(256, 64)       # Latent mean
        self.fc_logvar = nn.Linear(256, 64)   # Latent log-variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Input: [batch, 1, 28, 28]
        x = x.view(x.size(0), -1)   # [batch, 784]
        h = self.encoder(x)          # [batch, 256]
        mu = self.fc_mu(h)           # [batch, 64]
        logvar = self.fc_logvar(h)   # [batch, 64]
        z = self.reparameterize(mu, logvar)  # [batch, 64]
        out = self.decoder(z)        # [batch, 784]
        out = out.view(-1, 1, 28, 28)
        return out, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss
