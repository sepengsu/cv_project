import torch
import numpy as np
import os
import torch.nn as nn
class EarlyStopping():
    def __init__(self, patience=10, verbose=False, delta=0):
        '''
        patience (int): 얼마나 기다릴지
        verbose (bool): True일 경우 각 epoch의 loss 출력
        delta (float): 개선이 되었다고 인정되는 최소한의 loss
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss # validation loss가 작을수록 좋다고 가정

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
        
def model_delete(models, loss_functions, checkpoint_dir):
    '''
    이미 학습된 모델 + loss 조합을 models, loss_functions에서 제거
    checkpoint_dir/{model_name}_{loss_name}.pth 존재 여부로 판단
    '''
    models_to_delete = []
    models_not_to_delete = []
    for model_name in list(models.keys()):
        # 현재 모델이 사용 가능한 loss 목록
        if hasattr(models[model_name], 'T'):
            losses = list(loss_functions['diffusion'].keys())
        else:
            losses = list(loss_functions['reconstruction'].keys())
        
        # 해당 모델의 모든 loss 조합이 학습되었는지 확인
        all_ckpt_exist = all(
            os.path.exists(os.path.join(checkpoint_dir, f"{model_name}_{loss.replace('+','and')}.pth"))
            for loss in losses
        )

        if all_ckpt_exist:
            models_to_delete.append(model_name)
        else:
            models_not_to_delete.append(model_name)
    
    print(f"✅ {len(models_to_delete)} models and loss functions will be deleted: {models_to_delete}")
    print(f"✅ {len(models_not_to_delete)} models and loss functions will be kept: {models_not_to_delete}")
    # 실제 삭제
    for model_name in models_to_delete:
        del models[model_name]

    return models

def dir_clear(log_dir):
    '''
    log_dir: 로그를 저장할 디렉토리 경로
    하위 폴더 및 파일을 삭제
    '''
    if os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        print(f"✅ {log_dir} cleared.")

    else:
        print(f"❌ {log_dir} does not exist. No need to clear.")
    os.makedirs(log_dir, exist_ok=True)

from torchvision import transforms

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(0, self.std)
            return x + noise
        return x
    
mytrainsform= transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, augmentation_factor=1):
        '''
        dataset: 원본 데이터셋
        transform: 증강을 위한 transform
        augmentation_factor: 몇 배로 Augmentation (1 포함)
        '''
        self.dataset = dataset
        self.transform = transform
        self.augmentation_factor = augmentation_factor
        self.original_length = len(dataset)

    def __len__(self):
        # 전체 데이터 수 = 원본 * 배수
        return self.original_length * self.augmentation_factor

    def __getitem__(self, idx):
        # 원본 인덱스
        original_idx = idx % self.original_length
        x, y = self.dataset[original_idx]

        # factor == 1 이거나 첫 번째 패스는 원본 사용
        if self.augmentation_factor > 1 and idx >= self.original_length:
            if self.transform:
                x = self.transform(x)
        
        return x, y
