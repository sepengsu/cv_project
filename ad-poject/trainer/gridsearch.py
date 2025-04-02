import copy, os
from tqdm import trange
import torch
import torch.nn as nn
from .trainer import OneEpochTrainer,OneEpochTrainerFP16

class GridSearchTrainer(OneEpochTrainer):
    def __init__(self, models, criterions_dict, train_loader, val_loader,
                 n_epochs=50, patience=10, save_dir='./checkpoints',
                 lr=None, verbose=True, device=None):
        super().__init__(train_loader, val_loader, device)
        self.models = models
        self.reconstruction_criterions = criterions_dict['reconstruction']
        self.diffusion_criterions = criterions_dict['diffusion']
        self.n_epochs = n_epochs
        self.patience = patience
        self.save_dir = save_dir
        self.verbose = verbose
        self.lr = lr 
        os.makedirs(save_dir, exist_ok=True)

    def loss_candidates(self, model):
        if hasattr(model, 'T'):
            return list(self.diffusion_criterions.keys()), list(self.diffusion_criterions.values())
        elif hasattr(model, 'recon_loss_fn'):
            return list(self.reconstruction_criterions.keys()), list(self.reconstruction_criterions.values())
        else:
            return list(self.reconstruction_criterions.keys()), list(self.reconstruction_criterions.values())

    def print_combination(self):
        total_models = len(self.models)
        reconstruction_losses = len(self.reconstruction_criterions)
        diffusion_losses = len(self.diffusion_criterions)
        total_combinations = sum(len(self.loss_candidates(m)[0]) for m in self.models.values())

        print("=" * 50)
        print(f"Total Models: {total_models}")
        print(f"Reconstruction Losses: {reconstruction_losses}", end=' ')
        print(f"Diffusion Losses: {diffusion_losses}")
        print(f"Total Combinations: {total_combinations}")
        print("=" * 50)

    def run(self):
        self.print_combination()
        results = []

        for model_name, model_instance in self.models.items():
            loss_names, criterions = self.loss_candidates(model_instance)

            for loss_name, criterion in zip(loss_names, criterions):
                print(f'▶ Training [{model_name}] with [{loss_name}]')

                if hasattr(model_instance, 'T'):
                    model = model_instance.__class__(loss_fn=criterion).to(self.device)
                    criterion_in_trainer = None

                elif hasattr(model_instance, 'loss_function') or hasattr(model_instance, 'recon_loss_fn'):
                    model = model_instance.__class__(recon_loss_fn=criterion).to(self.device)
                    criterion_in_trainer = None

                else:
                    model = model_instance.to(self.device)
                    criterion_in_trainer = criterion.to(self.device) if criterion is not None else None

                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.n_epochs, eta_min=self.lr * 0.01
                )

                best_val_loss = float('inf')
                best_model = None
                best_epoch = 0
                early_stop_counter = 0

                for epoch in trange(self.n_epochs, desc=f"{model_name} | {loss_name}"):
                    train_loss = self.train_one_epoch(model, optimizer, criterion_in_trainer)
                    val_loss = self.validate_one_epoch(model, criterion_in_trainer)
                    scheduler.step()

                    if self.verbose:
                        print(f'[Epoch {epoch+1}/{self.n_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(model.state_dict())
                        best_epoch = epoch + 1
                        early_stop_counter = 0
                        if self.verbose:
                            print(f'>> Best Updated (Val Loss: {best_val_loss:.4f})')
                    else:
                        early_stop_counter += 1

                    if early_stop_counter >= self.patience:
                        if self.verbose:
                            print(f'>> Early Stopping at Epoch {epoch+1}')
                        break

                save_name = '' if loss_name == '-' else f'_{loss_name.replace("+", "and")}'
                save_path = f'{self.save_dir}/{model_name}{save_name}.pth'
                torch.save(best_model, save_path)
                print(f'>> Saved Best [{model_name}] + [{loss_name}] -> {save_path}')
                print(f'>> Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}')

                results.append({
                    "model": model_name,
                    "loss": loss_name,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "save_path": save_path
                })

                del model, optimizer, scheduler
                if criterion_in_trainer is not None:
                    del criterion_in_trainer
                torch.cuda.empty_cache()

        return results

# ==========================
# GridSearchTrainerFP16
# ==========================

class GridSearchTrainerFP16(OneEpochTrainerFP16):
    def __init__(self, models, criterions_dict, train_loader, val_loader,
                 n_epochs=50, patience=10, save_dir='./checkpoints',
                 lr=None, verbose=True, device=None):
        super().__init__(train_loader, val_loader, device)
        self.models = models
        self.reconstruction_criterions = criterions_dict['reconstruction']
        self.diffusion_criterions = criterions_dict['diffusion']
        self.n_epochs = n_epochs
        self.patience = patience
        self.save_dir = save_dir
        self.verbose = verbose
        self.lr = lr
        self.scaler = torch.amp.GradScaler('cuda')
        os.makedirs(save_dir, exist_ok=True)

    def loss_candidates(self, model):
        if hasattr(model, 'T'):
            return list(self.diffusion_criterions.keys()), list(self.diffusion_criterions.values())
        elif hasattr(model, 'recon_loss_fn'):
            return list(self.reconstruction_criterions.keys()), list(self.reconstruction_criterions.values())
        else:
            return list(self.reconstruction_criterions.keys()), list(self.reconstruction_criterions.values())

    def print_combination(self):
        total_models = len(self.models)
        reconstruction_losses = len(self.reconstruction_criterions)
        diffusion_losses = len(self.diffusion_criterions)
        total_combinations = sum(len(self.loss_candidates(m)[0]) for m in self.models.values())

        print("=" * 50)
        print(f"Total Models: {total_models}")
        print(f"Reconstruction Losses: {reconstruction_losses}", end=' ')
        print(f"Diffusion Losses: {diffusion_losses}")
        print(f"Total Combinations: {total_combinations}")
        print("=" * 50)

    def run(self):
        self.print_combination()
        results = []

        for model_name, model_instance in self.models.items():
            loss_names, criterions = self.loss_candidates(model_instance)

            for loss_name, criterion in zip(loss_names, criterions):
                print(f'▶ Training [{model_name}] with [{loss_name}] (FP16)')

                # ------------------------
                # Model build
                # ------------------------
                if hasattr(model_instance, 'T'):
                    model = model_instance.__class__(loss_fn=criterion).to(self.device)
                    criterion_in_trainer = None

                elif hasattr(model_instance, 'loss_function') or hasattr(model_instance, 'recon_loss_fn'):
                    model = model_instance.__class__(recon_loss_fn=criterion).to(self.device)
                    criterion_in_trainer = None

                else:
                    model = model_instance.to(self.device)
                    criterion_in_trainer = criterion.to(self.device) if criterion is not None else None

                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.n_epochs, eta_min=self.lr * 0.01
                )

                best_val_loss = float('inf')
                best_model = None
                best_epoch = 0
                early_stop_counter = 0

                # ------------------------
                # Training Loop (FP16)
                # ------------------------
                for epoch in trange(self.n_epochs, desc=f"{model_name} | {loss_name} (FP16)"):
                    train_loss = self.train_one_epoch(model, optimizer, criterion_in_trainer)
                    val_loss = self.validate_one_epoch(model, criterion_in_trainer)
                    scheduler.step()

                    if self.verbose:
                        print(f'[Epoch {epoch+1}/{self.n_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(model.state_dict())
                        best_epoch = epoch + 1
                        early_stop_counter = 0
                        if self.verbose:
                            print(f'>> Best Updated (Val Loss: {best_val_loss:.4f})')
                    else:
                        early_stop_counter += 1

                    if early_stop_counter >= self.patience:
                        if self.verbose:
                            print(f'>> Early Stopping at Epoch {epoch+1}')
                        break

                save_name = '' if loss_name == '-' else f'_{loss_name.replace("+", "and")}fp16'
                save_path = f'{self.save_dir}/{model_name}{save_name}.pth'
                torch.save(best_model, save_path)
                print(f'>> Saved Best [{model_name}] + [{loss_name}] -> {save_path}')
                print(f'>> Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}')

                results.append({
                    "model": model_name,
                    "loss": loss_name,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "save_path": save_path
                })

                del model, optimizer, scheduler
                if criterion_in_trainer is not None:
                    del criterion_in_trainer
                torch.cuda.empty_cache()

        return results
