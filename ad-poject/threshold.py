import numpy as np

class ThresholdSelector:
    def __init__(self, mode='percentile', percentile=0.95, k=3):
        """
        mode:
            - 'percentile': 기본 (상위 p% score를 threshold로)
            - 'mean_std' : mu + k * std
            # 향후 추가 가능:
            - 'kde'
            - 'adaptive'
        """
        self.mode = mode
        self.percentile = percentile
        self.k = k

    def fit(self, val_scores):
        if self.mode == 'percentile':
            self.threshold = np.percentile(val_scores, self.percentile * 100)
        elif self.mode == 'mean_std':
            mu = np.mean(val_scores)
            std = np.std(val_scores)
            self.threshold = mu + self.k * std
        else:
            raise NotImplementedError(f"Mode '{self.mode}' is not implemented yet.")
        return self.threshold

    def predict(self, scores):
        return (scores > self.threshold).astype(int)

    def get_threshold(self):
        return self.threshold
