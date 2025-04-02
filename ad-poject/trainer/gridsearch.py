import copy, os, gc
from tqdm import trange
import torch
import torch.nn as nn
from .trainer import OneEpochTrainer,OneEpochTrainerFP16
from .logger import LoggerMixin

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

class GridSearchTrainerFP16(OneEpochTrainerFP16, LoggerMixin):
    def __init__(self, models, criterions_dict, train_loader, val_loader,
                 n_epochs=50, patience=10, save_dir='./checkpoints',
                 lr=None, verbose=True, device=None, log_dir="./runs"):

        OneEpochTrainerFP16.__init__(self, train_loader, val_loader, device)
        LoggerMixin.__init__(self, log_dir=log_dir)

        self.models = models  # {model_name: model_class}
        self.reconstruction_criterions = criterions_dict['reconstruction']
        self.diffusion_criterions = criterions_dict['diffusion']
        self.n_epochs = n_epochs
        self.patience = patience
        self.save_dir = save_dir
        self.verbose = verbose
        self.lr = lr
        os.makedirs(save_dir, exist_ok=True)

    def loss_candidates(self, model_class):
        model_instance = model_class()
        if hasattr(model_instance, 'T'):
            return list(self.diffusion_criterions.items())
        else:
            return list(self.reconstruction_criterions.items())

    def print_combination(self):
        total_models = len(self.models)
        total_combinations = sum(len(self.loss_candidates(m)) for m in self.models.values())
        print("=" * 50)
        print(f"Total Models: {total_models}")
        print(f"Reconstruction Losses: {len(self.reconstruction_criterions)}", end=' ')
        print(f"Diffusion Losses: {len(self.diffusion_criterions)}")
        print(f"Total Combinations: {total_combinations}")
        print("=" * 50)

    def run(self):
        self.print_combination()
        results = []

        for model_name, model_class in self.models.items():
            for loss_name, criterion in self.loss_candidates(model_class):
                print(f'▶ Training [{model_name}] with [{loss_name}] (FP16)')
                model = model_class()
                if hasattr(model, 'T'):
                    model.loss_fn = criterion
                    criterion_in_trainer = None
                elif hasattr(model, 'loss_function') or hasattr(model, 'recon_loss_fn'):
                    model.recon_loss_fn = criterion
                    criterion_in_trainer = None
                else:
                    criterion_in_trainer = criterion.to(self.device) if criterion is not None else None
                model = model.to(self.device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.n_epochs, eta_min=self.lr * 0.01)

                best_val_loss, best_epoch, early_stop_counter = float('inf'), 0, 0
                best_model = None

                for epoch in trange(self.n_epochs, desc=f"{model_name} | {loss_name} (FP16)"):
                    train_loss = self.train_one_epoch(model, optimizer, criterion_in_trainer)
                    val_loss = self.validate_one_epoch(model, criterion_in_trainer)
                    scheduler.step()

                    self.log_losses(train_loss, val_loss, epoch)
                    self.log_gpu_usage(epoch)

                    if epoch % 10 == 0:
                        with torch.no_grad():
                            sample_x = next(iter(self.val_loader))[0].to(self.device)
                            output = model(sample_x)
                            if isinstance(output, tuple):
                                output = output[0]
                            self.log_images(sample_x, output, epoch)

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
                            print(f'>> Early Stopping at Epoch {epoch + 1}')
                        break

                for k in best_model:
                    best_model[k] = best_model[k].cpu()

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

                self._release_memory(model, optimizer, scheduler, criterion_in_trainer)

        self.close_logger()
        return results

    def _release_memory(self, model, optimizer, scheduler, criterion_in_trainer):
        model.cpu()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        if hasattr(scheduler, 'optimizer'):
            for state in scheduler.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
        if criterion_in_trainer is not None and hasattr(criterion_in_trainer, 'to'):
            criterion_in_trainer.cpu()

        del model, optimizer, scheduler
        if criterion_in_trainer is not None:
            del criterion_in_trainer
        torch.cuda.empty_cache()
