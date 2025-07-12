import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import nullcontext
from .stepping import StepFactory, fAnoGANStep, GANomalyStep

warmup_epochs = 5
class OneEpochTrainerFP16:
    def __init__(self, train_loader, val_loader, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.fp16 = self.device.type == "cuda"
        self.fp16 = False  # FP16을 사용하지 않도록 설정
        self.scaler = torch.amp.GradScaler() if self.fp16 else None
        self.step = None

    def train_one_epoch(self, epoch_count, model, optimizer_G, criterion, optimizer_D=None, scheduler=None):
        total_loss = 0
        model.train()
        pbar = tqdm(self.train_loader, desc="Train", leave=False)

        if self.step is None:
            self.step = StepFactory.create(model, self.device, self.scaler)

        for x, _ in pbar:
            x = x.to(self.device)
            optimizer_G.zero_grad()

            autocast_ctx = torch.autocast("cuda") if self.fp16 else nullcontext()
            with autocast_ctx:
                if isinstance(self.step, (fAnoGANStep, GANomalyStep)):
                    loss = self.step.forward(
                        model, x, criterion, is_train=True,
                        optimizer_D=optimizer_D, optimizer_G=optimizer_G,
                        warmup=(epoch_count < warmup_epochs)  # warmup_epochs는 상수로 정의
                    )
                else:
                    loss = self.step.forward(model, x, criterion, is_train=True, 
                                             optimizer_D=optimizer_D,
                                            optimizer_G=optimizer_G,
                                            warmup=(epoch_count < warmup_epochs))
                                             

            if not isinstance(self.step, (fAnoGANStep, GANomalyStep)):
                if self.fp16:
                    self.scaler.scale(loss).backward()

                    # ✅ 스케일된 step + 명시적 step 호출 → 경고 제거 핵심
                    self.scaler.step(optimizer_G)
                    optimizer_G.step()  # PyTorch가 step 인식하게끔 명시적으로 호출
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer_G.step()

                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate_one_epoch(self, epoch_count, model, criterion):
        total_loss = 0
        model.eval()
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        if self.step is None:
            self.step = StepFactory.create(model, self.device, self.scaler)

        with torch.no_grad():
            for x, _ in pbar:
                x = x.to(self.device)
                if isinstance(self.step, (fAnoGANStep, GANomalyStep)):
                    loss = self.step.forward(model, x, criterion, is_train=False, warmup=(epoch_count < 10))
                else:
                    loss = self.step.forward(model, x, criterion, is_train=False)

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)
