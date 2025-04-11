import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .stepping import StepFactory


class OneEpochTrainerFP16:
    def __init__(self, train_loader, val_loader, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler('cuda')  # FP16 scaler
        self.step = None  # Step 객체를 멤버로 등록

    def train_one_epoch(self, model, optimizer, criterion, is_train=True, optimizer_D=None):
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)

        if self.step is None:  # 최초 한 번만 생성
            self.step = StepFactory.create(model, self.device, self.scaler)

        for x, _ in pbar:
            x = x.to(self.device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss = self.step.forward(model, x, criterion, is_train=is_train, optimizer_D=optimizer_D)

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

        if self.step is None:
            self.step = StepFactory.create(model, self.device, self.scaler)

        with torch.no_grad():
            for x, _ in pbar:
                x = x.to(self.device)
                loss = self.step.forward(model, x, criterion, is_train=False, optimizer_D=None)
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)
