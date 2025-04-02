import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# CVAE
# -----------------------------
class CVAE(nn.Module):
    def __init__(self, latent_dim=64, recon_loss_fn=nn.MSELoss(reduction='sum')):
        super(CVAE, self).__init__()
        self.recon_loss_fn = recon_loss_fn
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        d = self.fc_decode(z).view(-1, 64, 7, 7)
        recon_x = self.decoder(d)
        return recon_x, mu, logvar

    def loss_function(self, x, recon_x, mu, logvar):
        recon_loss = self.recon_loss_fn(recon_x, x) / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_loss

# -----------------------------
# DeepCVAE
# -----------------------------
class DeepCVAE(nn.Module):
    def __init__(self, latent_dim=64, recon_loss_fn=nn.MSELoss(reduction='sum')):
        super(DeepCVAE, self).__init__()
        self.recon_loss_fn = recon_loss_fn
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        d = self.fc_decode(z).view(-1, 256, 4, 4)
        recon_x = self.decoder(d)
        return recon_x, mu, logvar

    def loss_function(self, x, recon_x, mu, logvar):
        recon_loss = self.recon_loss_fn(recon_x, x) / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_loss
