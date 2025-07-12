import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseStep:
    def __init__(self, device):
        self.device = device

    def forward(self, model, x, criterion=None, is_train=False, optimizer_D=None, optimizer_G=None, warmup=False):
        raise NotImplementedError("Subclass must implement this")


class AENormalStep(BaseStep):
    def forward(self, model, x, criterion=None, is_train=False, optimizer_D=None, optimizer_G=None, warmup=False):
        '''
        warmup: True일 경우--> criteraion은 항상 mse로 고정
        warmup: False일 경우--> criteraion은 model에 정의된 loss_function으로 고정
        '''
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        if warmup:
            criterion = nn.MSELoss(reduction='mean').to(self.device)

        return criterion(output, x)


class DiffusionStep(BaseStep):
    def forward(self, model, x, criterion=None, is_train=False, optimizer_D=None, optimizer_G=None, warmup=False):
        t = torch.randint(0, model.T, (x.size(0),), device=self.device)
        noise_pred, noise = model(x, t)
        return model.loss_function(noise_pred, noise)


class GANomalyStep(BaseStep):
    def __init__(self, device, scaler):
        super().__init__(device)
        self.scaler = scaler
        self.bce = nn.BCEWithLogitsLoss().to(device)

    def forward(self, model, x, criterion=None, is_train=False, optimizer_D=None, optimizer_G=None, warmup=False):
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

class fAnoGANStep(BaseStep):
    def __init__(self, device, scaler=None):
        super().__init__(device)
        self.scaler = scaler
        self.bce = nn.BCEWithLogitsLoss().to(device)

    def forward(self, model, x, criterion=None, is_train=False,
                optimizer_D=None, optimizer_G=None, warmup=False):
        # 1. Forward: Encode → Generate → Detach
        z = model.encode(x)
        x_hat = model.generate(z)
        x_hat_detach = x_hat.detach()
        z_hat = model.encode(x_hat_detach)

        # 2. Discriminator 학습
        if is_train and not warmup and optimizer_D is not None:
            d_real = model.discriminate(x)
            d_fake = model.discriminate(x_hat_detach)

            d_loss_real = self.bce(d_real, torch.ones_like(d_real))
            d_loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_loss_real + d_loss_fake) / 2

            optimizer_D.zero_grad()
            if self.scaler:
                self.scaler.scale(d_loss).backward()
                self.scaler.step(optimizer_D)
            else:
                d_loss.backward()
                optimizer_D.step()

        # 3. Generator 학습 (Adversarial + Reconstruction + Latent)
        d_fake_for_G = model.discriminate(x_hat)
        adv_loss = self.bce(d_fake_for_G, torch.ones_like(d_fake_for_G))
        recon_loss = criterion(x_hat, x)
        latent_loss = F.l1_loss(z_hat, z)
        total_loss = recon_loss + 0.1 * adv_loss + 0.1 * latent_loss

        if is_train and optimizer_G is not None:
            optimizer_G.zero_grad()
            scaled_loss = self.scaler.scale(total_loss) if self.scaler else total_loss
            scaled_loss.backward()

            if warmup:
                # step을 하되, 실제로 학습되지 않도록 dummy step (scaler 내부 등록용)
                if self.scaler:
                    self.scaler.step(optimizer_G)
                    self.scaler.update()
            else:
                if self.scaler:
                    self.scaler.step(optimizer_G)
                    self.scaler.update()
                else:
                    optimizer_G.step()



        # ⚠️ 디버깅용 로그 (원할 시 주석 해제)
        # if is_train:
        #     print(f"[fAnoGAN Step] recon: {recon_loss.item():.4f}, adv: {adv_loss.item():.4f}, latent: {latent_loss.item():.4f}"
        # for n, p in model.generator.named_parameters():
        #     print(n, p.requires_grad)

        return total_loss



class fAnoVAEStep(BaseStep):
    def __init__(self, device):
        super().__init__(device)

    def forward(self, model, x, criterion=None, is_train=False, optimizer_D=None):
        x_hat, mu, logvar, z, z_hat_mu, _ = model(x)
        return model.loss_function(x, x_hat, mu, logvar, z, z_hat_mu)

class StepFactory:
    @staticmethod
    def create(model, device, scaler=None):
        if hasattr(model, 'name') and model.name == 'fAnoVAE':
            return fAnoVAEStep(device)
        elif hasattr(model, 'name') and model.name.startswith('fAnoGAN'):
            return fAnoGANStep(device, scaler)
        elif hasattr(model, 'T'):
            return DiffusionStep(device)
        elif hasattr(model, 'discriminator'):
            return GANomalyStep(device, scaler)
        elif hasattr(model,'name') and model.name in ['ViTAE', 'LGViTAE']:
            return AENormalStep(device)
        else:
            return AENormalStep(device)


