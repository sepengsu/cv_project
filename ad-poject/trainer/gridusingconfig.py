import torch
import os
import copy
from tqdm import trange
from torch.utils.data import DataLoader
from .trainer import OneEpochTrainerFP16
from .logger import LoggerMixin, GPUUsageLoggerMixin
from .stepping import StepFactory
from eval_module.eval_show import EvalDataset
from loss.losses import FlexibleLoss
from .initing import weights_init

class GridSearchTrainerUsingConfig(OneEpochTrainerFP16, LoggerMixin, GPUUsageLoggerMixin):
    def __init__(self, model_configs, train_dataset, val_dataset,
                 n_epochs=50, patience=10, save_root='./checkpoints',
                 verbose=True, device=None, log_root="./runs"):

        self.model_configs = model_configs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_epochs = n_epochs
        self.patience = patience
        self.verbose = verbose
        self.save_root = save_root
        self.log_root = log_root
        self.show_dataset = EvalDataset()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(save_root, exist_ok=True)

    def run(self):
        results = []

        for model_name, config in self.model_configs.items():
            model_class = config["class"]
            lr = config["lr"]
            batch_size = config["batch_size"]
            loss_names = config["losses"]
            loss_weights = config.get("loss_weights", None)

            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

            for loss_name in loss_names:
                model = model_class().to(self.device)
                model.apply(weights_init) # ✅ 모델 초기화

                # ✅ loss_weights 기반으로 FlexibleLoss 초기화
                criterion = FlexibleLoss(mode=loss_name, loss_weights=loss_weights, reduction='mean')

                LoggerMixin.__init__(self, log_dir=os.path.join(self.log_root, model_name, loss_name))
                GPUUsageLoggerMixin.__init__(self)

                trainer = OneEpochTrainerFP16(train_loader, val_loader, self.device)


                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=lr * 0.1)

                best_val_loss = float('inf')
                best_model = None
                best_epoch = 0
                early_stop_counter = 0

                self.start_gpu_monitor()

                for epoch in trange(self.n_epochs, desc=f"{model_name} | {loss_name}"):
                    # ✅ scheduler를 trainer 내부로 전달
                    train_loss = trainer.train_one_epoch(epoch, model, optimizer, criterion, scheduler=scheduler)
                    val_loss = trainer.validate_one_epoch(epoch, model, criterion)
                    self.log_losses(train_loss, val_loss, epoch)
                    if epoch % 10 == 0:
                        with torch.no_grad():
                            sample_x, label = next(iter(torch.utils.data.DataLoader(self.show_dataset, batch_size=8)))
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
                            print(f'>> Early stopping at epoch {epoch+1}')
                        break

                save_path = self.save(model_name, loss_name, best_model, best_val_loss, best_epoch)

                results.append({
                    "model": model_name,
                    "loss": loss_name,
                    "val_loss": best_val_loss,
                    "epoch": best_epoch,
                    "batch_size": batch_size,
                    "lr": lr,
                    "save_path": save_path,
                    "gpu_peak": self.gpu_peak_usage
                })

                self.stop_gpu_monitor()
                self.step = None
                model.cpu()
                del model, optimizer, scheduler, trainer
                torch.cuda.empty_cache()
                

        self.close_logger()
        return results
    
    def save(self, model_name, loss_name, best_model, best_val_loss, best_epoch):
        for k in best_model:
            best_model[k] = best_model[k].cpu()
        save_name = '' if loss_name == '-' else f'_{loss_name.replace("+", "and")}fp16'
        save_path = f'{self.save_root}/{model_name}{save_name}.pth'
        torch.save(best_model, save_path)
        if self.verbose:
            print(f'>> Model [{model_name}] + Loss [{loss_name}] Saved to {save_path}')

        print(f'>> Saved Best [{model_name}] + [{loss_name}] -> {save_path}')
        print(f'>> Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch} and GPU Usage: {self.gpu_peak_usage:.2f} MB')
        return save_path