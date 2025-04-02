import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score
from tqdm import tqdm

class Evaluator:
    def __init__(self, checkpoint_dir, val_loader, test_loader, device=None, percentile=0.95, save_dir='./eval_results'):
        self.checkpoint_dir = checkpoint_dir
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.percentile = percentile
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def load_model(self, model_name):
        """모델 이름으로부터 instance 생성"""
        model_class = get_model_classes()[model_name]
        model = model_class().to(self.device)
        return model

    def compute_score(self, model, x):
        """Diffusion → noise mse / AE, VAE, etc → recon mse"""
        if hasattr(model, 'T'):
            t = torch.randint(0, model.T, (x.size(0),), device=x.device)
            noise_pred, noise = model(x, t)
            return F.mse_loss(noise_pred, noise, reduction='none').view(x.size(0), -1).mean(dim=1)
        else:
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            return F.mse_loss(output, x, reduction='none').view(x.size(0), -1).mean(dim=1)

    def get_scores(self, loader, model):
        """Validation 또는 Test set에 대한 score 계산"""
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Scoring"):
                x = x.to(self.device)
                score = self.compute_score(model, x)
                scores.append(score.cpu())
                labels.append(y.cpu())
        return torch.cat(scores).numpy(), torch.cat(labels).numpy()

    def run(self):
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pth")]
        results = []

        for ckpt in checkpoint_files:
            print(f"\n▶ Evaluating {ckpt}")

            # 안전하게 loss_name에 _가 여러 개 들어가도 마지막만 split
            model_name, loss_name = ckpt[:-4].rsplit("_", 1)

            model = self.load_model(model_name)
            model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, ckpt)))

            # --------------------------
            # Threshold 계산 (Validation)
            # --------------------------
            val_scores, _ = self.get_scores(self.val_loader, model)
            threshold = np.percentile(val_scores, self.percentile * 100)
            print(f" >> Threshold ({self.percentile*100:.0f}%) = {threshold:.4f}")

            # --------------------------
            # Test 평가
            # --------------------------
            test_scores, test_labels = self.get_scores(self.test_loader, model)
            test_labels = (test_labels != 2).astype(int)  # class 2 = normal
            preds = (test_scores > threshold).astype(int)

            auc_score = roc_auc_score(test_labels, test_scores)
            precision, recall, _ = precision_recall_curve(test_labels, test_scores)
            pr_auc = auc(recall, precision)
            f1 = f1_score(test_labels, preds)
            acc = accuracy_score(test_labels, preds)

            print(f"ROC-AUC: {auc_score:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | ACC: {acc:.4f}")

            results.append({
                "checkpoint": ckpt,
                "threshold": threshold,
                "roc_auc": auc_score,
                "pr_auc": pr_auc,
                "f1": f1,
                "acc": acc
            })

        df = pd.DataFrame(results)
        return df
