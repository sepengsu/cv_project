import torch
import multiprocessing as mp
import torch.nn.functional as F
import copy, os
from tqdm import tqdm

class GridSearchTrainer:
    def __init__(self, models, criterions, train_loader, val_loader, n_epochs=50, patience=10, save_dir='./checkpoints', verbose=True, device=None):
        """
        models: {"model_name": model()}
        criterions: {"loss_name": loss_function}
        """
        self.models = models
        self.criterions = criterions
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.patience = patience
        self.save_dir = save_dir
        self.verbose = verbose
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(save_dir, exist_ok=True)

    def train_one_epoch(self, model, optimizer, criterion):
        model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for x, _ in pbar:
            x = x.to(self.device)
            output = model(x)
            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def validate_one_epoch(self, model, criterion):
        model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for x, _ in pbar:
                x = x.to(self.device)
                output = model(x)
                loss = criterion(output, x)
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        return total_loss / len(self.val_loader)

    def run(self):
        results = []

        for model_name, model in self.models.items():
            for loss_name, criterion in self.criterions.items():
                print(f'▶ Training [{model_name}] with [{loss_name}]')

                model = model.to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                best_val_loss = float('inf')
                best_model = None
                early_stop_counter = 0

                for epoch in range(self.n_epochs):
                    train_loss = self.train_one_epoch(model, optimizer, criterion)
                    val_loss = self.validate_one_epoch(model, criterion)

                    if self.verbose:
                        print(f'[Epoch {epoch+1}/{self.n_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(model.state_dict())
                        early_stop_counter = 0
                        if self.verbose:
                            print(f'>> Best Updated (Val Loss: {best_val_loss:.4f})')
                    else:
                        early_stop_counter += 1

                    if early_stop_counter >= self.patience:
                        if self.verbose:
                            print(f'>> Early Stopping at Epoch {epoch+1}')
                        break

                clean_loss_name = loss_name.replace("+", "and")
                save_path = f'{self.save_dir}/{model_name}_{clean_loss_name}.pth'
                torch.save(best_model, save_path)
                print(f'>> Saved Best [{model_name}] + [{loss_name}] -> {save_path}')

                results.append({
                    "model": model_name,
                    "loss": loss_name,
                    "best_val_loss": best_val_loss,
                    "save_path": save_path
                })

        return results


class MultiModelTrainer:
    def __init__(self, models, criterions, train_loader, val_loader, 
                 n_epochs=50, patience=10, save_dir='./checkpoints', 
                 device=None, verbose=True, max_workers=3):
        self.models = models
        self.criterions = criterions
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.patience = patience
        self.save_dir = save_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.max_workers = max_workers  # ✅ 추가

    def _train_worker(self, model_name, model, loss_name, criterion):
        print(f"[Process] {model_name} + {loss_name} Start")
        trainer = GridSearchTrainer(
            models={model_name: model},
            criterions={loss_name: criterion},
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            n_epochs=self.n_epochs,
            patience=self.patience,
            save_dir=self.save_dir,
            device=self.device,
            verbose=self.verbose
        )
        trainer.run()
        print(f"[Process] {model_name} + {loss_name} Done")

    def run(self):
        jobs = []
        active_jobs = []

        for model_name, model in self.models.items():
            for loss_name, criterion in self.criterions.items():
                p = mp.Process(target=self._train_worker, args=(model_name, model, loss_name, criterion))
                p.start()
                active_jobs.append(p)

                # ✅ max_workers 초과 시 기다림
                if len(active_jobs) >= self.max_workers:
                    for job in active_jobs:
                        job.join()
                    active_jobs.clear()

        # 남은 작업 마저 기다림
        for job in active_jobs:
            job.join()

        print(">> All model x loss experiments completed.")