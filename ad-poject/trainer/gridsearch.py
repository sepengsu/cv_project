import copy, os, gc
from tqdm import trange
import torch
import torch.nn as nn
from .trainer import OneEpochTrainerFP16
from .logger import LoggerMixin, GPUUsageLoggerMixin
from eval_module.eval_show import EvalDataset
from .stepping import StepFactory, fAnoGANStep, GANomalyStep

class LossAssignmentHelper:
    @staticmethod
    def assign(model, criterion, device):
        if hasattr(model, 'name') and model.name == 'fAnoVAE':
            return None
        elif hasattr(model, 'name') and model.name in ['fAnoGAN', 'fAnoGAN_v2', 'fAnoGAN_v3', 'fAnoGAN_AllInOne']:
            return criterion.to(device)
        elif hasattr(model, 'T'):
            model.loss_fn = criterion
            return None
        elif hasattr(model, 'loss_function') or hasattr(model, 'recon_loss_fn'):
            model.recon_loss_fn = criterion
            return None       
        else:
            return criterion.to(device) if criterion is not None else None

class GridSearchTrainerFP16(OneEpochTrainerFP16, LoggerMixin, GPUUsageLoggerMixin):
    def __init__(self, models, criterions_dict, train_loader, val_loader,
                 n_epochs=50, patience=10, save_dir='./checkpoints',
                 lr=None, verbose=True, device=None, log_dir="./runs"):
        
        OneEpochTrainerFP16.__init__(self, train_loader, val_loader, device)
        self.models = models
        self.reconstruction_criterions = criterions_dict['reconstruction']
        self.diffusion_criterions = criterions_dict['diffusion']
        self.n_epochs = n_epochs
        self.patience = patience
        self.save_dir = save_dir
        self.verbose = verbose
        self.lr = lr
        self.log_dir = log_dir
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.show_dataset = EvalDataset()
        os.makedirs(save_dir, exist_ok=True)

    def loss_candidates(self, model_class):
        model_instance = model_class()
        if hasattr(model_instance, 'T') or (hasattr(model_instance, 'name') and 
            model_instance.name =="SSIMAE"):
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

    def run(self):
        self.print_combination()
        results = []

        for model_name, model_class in self.models.items():
            lr =self.lr
                
            for loss_name, criterion in self.loss_candidates(model_class):
                print(f'▶ Training [{model_name}] with [{loss_name}] (FP16)')

                model = model_class()
                LoggerMixin.__init__(self, log_dir=os.path.join(self.log_dir, model_name, loss_name))
                GPUUsageLoggerMixin.__init__(self)

                criterion_in_trainer = LossAssignmentHelper.assign(model, criterion, self.device)
                model = model.to(self.device)
                # Generator Optimizer 설정 (모델 이름별로 분기)
                if model_name.startswith("fAnoGAN"):
                    optimizer_G = torch.optim.AdamW(
                        list(model.encoder.parameters()) + list(model.generator.parameters()),
                        lr=lr, weight_decay=1e-4)
                    optimizer_D = torch.optim.AdamW(model.discriminator.parameters(), lr=lr*0.1, weight_decay=1e-4)
                else:
                    optimizer_G = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                    optimizer_D = None
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_G, T_max=self.n_epochs, eta_min=lr*0.1
                )

                best_val_loss, best_epoch, early_stop_counter = float('inf'), 0, 0
                best_model = None

                self.start_gpu_monitor()

                for epoch in trange(self.n_epochs, desc=f"{model_name} | {loss_name} (FP16)"):
                    train_loss = self.train_one_epoch(
                        epoch,model, optimizer_G, criterion_in_trainer,
                        optimizer_D=optimizer_D
                    )
                    val_loss = self.validate_one_epoch(epoch,model, criterion_in_trainer)

                    # Scheduler step only if generator optimizer was used
                    if not isinstance(self.step, (fAnoGANStep, GANomalyStep)):
                        scheduler.step()

                    if self.verbose:
                        print(f'>> Epoch [{epoch + 1}/{self.n_epochs}] - '
                              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                    # epoch 당 step 클래스를 초기화 
                    self.step = None
                    
                    if epoch % 10 == 0:
                        with torch.no_grad():
                            sample_x, label = next(iter(torch.utils.data.DataLoader(self.show_dataset, batch_size=6)))
                            sample_x = sample_x.to(self.device)
                            if hasattr(model, 'T'):
                                t = torch.randint(0, model.T, (sample_x.size(0),), device=sample_x.device)
                                output = model(sample_x, t)
                            else:
                                output = model(sample_x)
                            if isinstance(output, tuple):
                                output = output[0]
                            self.log_images(sample_x, label, output, epoch)

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

                # Save the best model
                save_path = self.save(model_name, loss_name, best_model, best_val_loss, best_epoch)
                self.stop_gpu_monitor()


                results.append({
                    "model": model_name,
                    "loss": loss_name,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "gpu_peak_usage(MB)": self.gpu_peak_usage,
                    "save_path": save_path
                })

                self._release_memory(model, optimizer_G, scheduler, criterion_in_trainer)

        self.close_logger()
        return results

    def save(self, model_name, loss_name, best_model, best_val_loss, best_epoch):
        for k in best_model:
            best_model[k] = best_model[k].cpu()
        save_name = '' if loss_name == '-' else f'_{loss_name.replace("+", "and")}fp16'
        save_path = f'{self.save_dir}/{model_name}{save_name}.pth'
        torch.save(best_model, save_path)
        if self.verbose:
            print(f'>> Model [{model_name}] + Loss [{loss_name}] Saved to {save_path}')

        print(f'>> Saved Best [{model_name}] + [{loss_name}] -> {save_path}')
        print(f'>> Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch} and GPU Usage: {self.gpu_peak_usage:.2f} MB')
        return save_path
