import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

# ✅ Light noise function
def add_gaussian_noise(x, std=0.01):
    return x + std * torch.randn_like(x)

class PulloverAugmentedDatasetFromTrainset(Dataset):
    def __init__(self, trainset, label_for_pullover=2, n=2):
        self.label_for_pullover = label_for_pullover

        # ✅ Augmentation은 Tensor 기준으로만 적용
        self.augment_transform = T.Compose([
            T.RandomCrop(28, padding=1, padding_mode='reflect'),  
            T.Lambda(add_gaussian_noise)
        ])

        # ✅ images와 labels 준비
        self.images = []
        self.labels = []

        for img, label in trainset:
            if label == self.label_for_pullover:
                if isinstance(img, torch.Tensor):
                    img = img.squeeze(0).numpy()  # 2D로 만듦
                self.images.append(img)
                self.labels.append(label)

        self.images = np.stack(self.images)  # (N, 28, 28)
        self.labels = np.array(self.labels)
        self.total_len = len(self.images) * n

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        real_idx = idx % len(self.images)
        img = self.images[real_idx]
        label = self.labels[real_idx]

        # ✅ numpy → torch.tensor 변환
        img = torch.tensor(img, dtype=torch.float32)

        if img.ndim == 2:
            img = img.unsqueeze(0)  # (1, 28, 28)

        # ✅ manual normalization
        if img.max() > 1.5:
            img = img / 255.0

        # ✅ 증강 적용 여부
        if idx >= len(self.images):
            img = self.augment_transform(img)

        return img, label

# ✅ DataLoader용 wrapper
def get_pullover_augmented_trainset(trainset, n=2):
    dataset = PulloverAugmentedDatasetFromTrainset(trainset, n=n)
    return dataset
