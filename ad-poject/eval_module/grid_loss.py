# Updated GridLossEvaluator with fallback protection and empty loss result guard
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score
from tqdm import tqdm
from .scoring import AnomalyScoreComputer
from .ploting import plot_and_save_score_distribution, plot_test_score_distribution

class GridLossEvaluator:
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        loss_fns: dict,  # {"loss_name": loss_fn (reduction='none')}
        model: torch.nn.Module,
        model_name: str = None,
        device=None,
        plot_dir="./score_plots",
        percentile=0.95,
        path = None,
        inference_dir = './inference_dir'
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fns = loss_fns
        self.model = model.to(device)
        self.model_name = model_name if model_name else model.__class__.__name__
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.percentile = percentile
        self.plot_dir =plot_dir
        self.path = path
        self.inference_dir = inference_dir

    def compute_score(self, model, x, loss_fn):
        score_computer = AnomalyScoreComputer(model, loss_fn, self.device)
        return score_computer.compute(x)

    def get_scores(self, loader, model, loss_fn, loss_name=""):
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                try:
                    score = self.compute_score(model, x, loss_fn)
                    if torch.isnan(score).any() or torch.isinf(score).any():
                        continue
                    scores.append(score.cpu())
                    labels.append(y.cpu())
                except Exception as e:
                    print(f"[❌ Error] Failed to compute score for {loss_name}: {e}")
        if not scores:
            return np.array([]), np.array([])
        return torch.cat(scores).numpy(), torch.cat(labels).numpy()

    def run(self):
        all_results = []
        model = self.model

        for loss_name, loss_fn in self.loss_fns.items():
            loss_fn = loss_fn.to(self.device)
            loss_fn.reduction = 'none'  # Ensure no reduction for score computation
            train_scores, _ = self.get_scores(self.train_loader, model, loss_fn, loss_name)
            val_scores, _ = self.get_scores(self.val_loader, model, loss_fn, loss_name)
            if len(val_scores) == 0:
                print(f"[‼️ Skip] No valid validation scores for {loss_name} → skipping this loss function")
                continue

            threshold = np.percentile(val_scores, self.percentile * 100)

            test_scores, test_labels = self.get_scores(self.test_loader, model, loss_fn, loss_name)
            if len(test_scores) == 0 or len(test_labels) == 0:
                print(f"[‼️ Skip] No valid test scores for {loss_name} → skipping this loss function")
                continue

            # ✅ 이진 라벨 처리 (2: 정상 → 0 / 4,6: 비정상 → 1)
            test_labels = (test_labels != 2).astype(int)

            # ✅ (선 적용) 다차원 score를 평균 처리 (sample-wise anomaly score)
            if test_scores.ndim > 1:
                test_scores = test_scores.reshape(test_scores.shape[0], -1).mean(axis=1)  # (B,)

            # ✅ threshold 적용 → 이진 예측
            preds = (test_scores > threshold).astype(int)

            # ✅ 개수 출력용 통계
            unique_labels, count_labels = np.unique(test_labels, return_counts=True)
            unique_preds, count_preds = np.unique(preds, return_counts=True)

            label_dist = {int(k): int(v) for k, v in zip(unique_labels, count_labels)}
            pred_dist = {int(k): int(v) for k, v in zip(unique_preds, count_preds)}

            print(f"[정상 개수, 예측 정상개수] {label_dist}, [비정상 개수, 예측 비정상개수] {pred_dist}")

            # ✅ 시각화
            plot_and_save_score_distribution(
                train_scores, val_scores, test_scores, threshold,
                loss_name, model_name=self.model_name, plot_dir=self.plot_dir,
                test_labels=test_labels  # ✅ 추가됨
            )
            plot_test_score_distribution(
                test_scores=test_scores, test_labels=test_labels, threshold=threshold,
                loss_name=loss_name, model_name=self.model_name, plot_dir=self.inference_dir
            )
            try:
                # ✅ Score 기반 metric
                auc_score = roc_auc_score(test_labels, test_scores)
                precision, recall, _ = precision_recall_curve(test_labels, test_scores)
                pr_auc = auc(recall, precision)

                # ✅ Binary classification metric
                f1 = f1_score(test_labels, preds)
                acc = accuracy_score(test_labels, preds)

            except Exception as e:
                print(f"[❌ Metric Error] {loss_name}: {e}")
                continue


            print(f" ROC-AUC: {auc_score:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | ACC: {acc:.4f} | Threshold: {threshold:.4f}")
            print("-" * 50)

            all_results.append({
                'model_name': model.__class__.__name__,
                "loss_type": loss_name,
                "threshold": threshold,
                "roc_auc": auc_score,
                "pr_auc": pr_auc,
                "f1": f1,
                "acc": acc,
                'num_normal': int(pred_dist.get(0, 0)),
                'num_abnormal': int(pred_dist.get(1, 0)),
                'path': self.path,
            })


        return pd.DataFrame(all_results) if all_results else pd.DataFrame(columns=[
            'model_name', 'loss_type', 'threshold', 'roc_auc', 'pr_auc', 'f1', 'acc', 'num_normal', 'num_abnormal'
        ])
