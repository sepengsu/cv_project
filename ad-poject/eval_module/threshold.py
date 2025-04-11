import numpy as np
import torch
from abc import ABC, abstractmethod
from scipy.stats import gaussian_kde, genpareto

# 공통 처리 함수
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


# ========================
# Base Abstract Class
# ========================
class ThresholdSelector(ABC):
    def __init__(self):
        self.threshold = None

    @abstractmethod
    def fit(self, val_scores):
        pass

    @abstractmethod
    def predict(self, scores):
        pass

    def get_threshold(self):
        return self.threshold


# ========================
# Percentile Threshold
# ========================
class PercentileThreshold(ThresholdSelector):
    def __init__(self, percentile=0.95):
        super().__init__()
        self.percentile = percentile

    def fit(self, val_scores):
        val_scores = to_numpy(val_scores)
        self.threshold = np.percentile(val_scores, self.percentile * 100)
        return self.threshold

    def predict(self, scores):
        scores = to_numpy(scores)
        return (scores > self.threshold).astype(int)


# ========================
# Mean + K*Std Threshold
# ========================
class MeanStdThreshold(ThresholdSelector):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def fit(self, val_scores):
        val_scores = to_numpy(val_scores)
        mu = np.mean(val_scores)
        std = np.std(val_scores)
        self.threshold = mu + self.k * std
        return self.threshold

    def predict(self, scores):
        scores = to_numpy(scores)
        return (scores > self.threshold).astype(int)


# ========================
# KDE Threshold
# ========================
class KDEThreshold(ThresholdSelector):
    def __init__(self, min_prob=1e-4):
        super().__init__()
        self.min_prob = min_prob

    def fit(self, val_scores):
        val_scores = to_numpy(val_scores)
        self.kde = gaussian_kde(val_scores)
        xs = np.linspace(min(val_scores), max(val_scores), 1000)
        probs = self.kde(xs)
        idx = np.where(probs < self.min_prob)[0]
        self.threshold = xs[idx[0]] if len(idx) > 0 else max(val_scores)
        return self.threshold

    def predict(self, scores):
        scores = to_numpy(scores)
        return (scores > self.threshold).astype(int)


# ========================
# EVT (GPD Tail) Threshold
# ========================
class EVTThreshold(ThresholdSelector):
    def __init__(self, tail_percentile=0.90, p_value=1e-3):
        super().__init__()
        self.tail_percentile = tail_percentile
        self.p_value = p_value

    def fit(self, val_scores):
        val_scores = to_numpy(val_scores)
        base = np.percentile(val_scores, self.tail_percentile * 100)
        excesses = val_scores[val_scores > base] - base
        c, loc, scale = genpareto.fit(excesses)
        self.threshold = base + genpareto.ppf(1 - self.p_value, c, loc=loc, scale=scale)
        return self.threshold

    def predict(self, scores):
        scores = to_numpy(scores)
        return (scores > self.threshold).astype(int)


# ========================
# Factory
# ========================
class ThresholdFactory:
    def __init__(self):
        self.registry = {
            'percentile': PercentileThreshold,
            'mean_std': MeanStdThreshold,
            'kde': KDEThreshold,
            'evt': EVTThreshold,
        }

    def create(self, mode, **kwargs):
        if mode not in self.registry:
            raise ValueError(f"Unknown threshold mode: {mode}")
        return self.registry[mode](**kwargs)

    def create_multiple(self, modes_with_args):
        selectors = {}
        for mode, kwargs in modes_with_args.items():
            selectors[mode] = self.create(mode, **kwargs)
        return selectors
