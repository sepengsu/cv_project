import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

class PulloverAugmentedDatasetFromTrainset(Dataset):
    def __init__(self, trainset, label_for_pullover=2, n=2):
        self.label_for_pullover = label_for_pullover
        
        self.augment_transform = T.Compose([
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
        
        # Pullover(2번)만 추출
        self.images = []
        self.labels = []
        for img, label in trainset:
            if label == self.label_for_pullover:
                if isinstance(img, torch.Tensor):
                    img = img.squeeze(0).numpy()  # (28,28)로 변환
                self.images.append(img)
                self.labels.append(label)
        
        self.images = np.stack(self.images)  # (N, 28, 28)
        self.labels = np.array(self.labels)
        
        self.total_len = len(self.images) * n  # 2배로

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        real_idx = idx % len(self.images)
        img = self.images[real_idx]
        label = self.labels[real_idx]

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # (1,28,28)

        if idx >= len(self.images):
            img = self.augment_transform(img)  # ❗ squeeze(0) 제거

        return img, label


# ✅ DataLoader 생성 함수
def get_pullover_augmented_trainset(trainset,n=2):
    dataset = PulloverAugmentedDatasetFromTrainset(trainset, n=n)
    return dataset
