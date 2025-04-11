import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score
from tqdm import tqdm


class GridLossEvaluator:
    def __init__(
        self,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        loss_fns: dict,  # {"loss_name": loss_fn (reduction='none')}
        model: torch.nn.Module,
        device=None,
        percentile=0.95,
    ):
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fns = loss_fns  # {"loss_name": loss_fn (reduction='none')}
        self.model = model.to(device)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.percentile = percentile

    def compute_score(self, model, x, loss_fn):
        # Case 1: 모델이 reconstruct() 제공 → Autoencoder-style 평가
        if hasattr(model, "reconstruct"):
            x_recon = model.reconstruct(x)
            score = loss_fn(x_recon, x)

        # Case 2: 모델이 T 속성을 갖고 있고 forward는 noise 예측 기반 (SimpleDDPM)
        elif hasattr(model, "T"):
            # fallback 처리: noise → noise 평가 (최종 평가 목적에는 부적합)
            t = torch.randint(0, model.T, (x.size(0),), device=self.device)
            noise_pred, noise = model(x, t)
            score = F.mse_loss(noise_pred, noise, reduction='none')

        # Case 3: 일반 autoencoder 구조 (DiffusionUNet 포함)
        else:
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]

            score = loss_fn(output, x)

        return score.view(score.size(0), -1).mean(dim=1)



    def get_scores(self, loader, model, loss_fn, loss_name=""):
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Scoring ({loss_name})"):
                x = x.to(self.device)
                score = self.compute_score(model, x, loss_fn)
                scores.append(score.cpu())
                labels.append(y.cpu())
        return torch.cat(scores).numpy(), torch.cat(labels).numpy()

    def run(self):
        all_results = []
        model = self.model

        for loss_name, loss_fn in self.loss_fns.items():
            print(f">> Using loss: {loss_name}")

            val_scores, _ = self.get_scores(self.val_loader, model, loss_fn, loss_name)
            threshold = np.percentile(val_scores, self.percentile * 100)
            print(f"Threshold: {threshold:.4f}")

            test_scores, test_labels = self.get_scores(self.test_loader, model, loss_fn, loss_name)
            test_labels = (test_labels != 2).astype(int)
            preds = (test_scores > threshold).astype(int)

            auc_score = roc_auc_score(test_labels, test_scores)
            precision, recall, _ = precision_recall_curve(test_labels, test_scores)
            pr_auc = auc(recall, precision)
            f1 = f1_score(test_labels, preds)
            acc = accuracy_score(test_labels, preds)

            print(f" ROC-AUC: {auc_score:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | ACC: {acc:.4f}")

            all_results.append({
                'model_name': model.__class__.__name__,
                "loss_type": loss_name,
                "threshold": threshold,
                "roc_auc": auc_score,
                "pr_auc": pr_auc,
                "f1": f1,
                "acc": acc
            })

        return pd.DataFrame(all_results)
