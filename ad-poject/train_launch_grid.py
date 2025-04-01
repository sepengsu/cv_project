
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    import torch, torchvision
    import torch.nn.functional as F
    from torch import nn, optim
    from torchvision import transforms, datasets


    # Computational device
    # Device will be set to GPU if it is available.(you should install valid Pytorch version with CUDA. Otherwise, it will be computed using CPU)
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("Using Device:", DEVICE)

    # Fashion MNIST dataset
    trainset = datasets.FashionMNIST(
        root      = './.data/', train = True,
        download  = True,
        transform = transforms.ToTensor())
    testset = datasets.FashionMNIST(
        root      = './.data/', train     = False,
        download  = True,
        transform = transforms.ToTensor())
    
    SELECT_NORMAL = 2 # Set 2 class as train dataset.
    trainset.data = trainset.data[trainset.targets == SELECT_NORMAL]
    trainset.targets = trainset.targets[trainset.targets == SELECT_NORMAL] # Set 2 class as train dataset.

    test_label = [2,4,6] # Define actual test class that we use
    actual_testdata = torch.isin(testset.targets, torch.tensor(test_label))
    testset.data = testset.data[actual_testdata]
    testset.targets = testset.targets[actual_testdata]

    test_loader = torch.utils.data.DataLoader(
        dataset     = testset, batch_size  = 1,
        shuffle     = False,num_workers = 2)

    train_data_size = len(trainset)
    test_data_size = len(testset)

    print("Train data size:", train_data_size, "Test data size:", test_data_size)

    class GaussianNoise(nn.Module):
        def __init__(self, std=0.1):
            super().__init__()
            self.std = std

        def forward(self, x):
            if self.training:
                noise = x.data.new(x.size()).normal_(0, self.std)
                return x + noise
            return x
        
    # 몇 배로 Augmentation을 할 것인지 알려주면 해당 배수만큼 Augmentation을 수행하는 클래스
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        GaussianNoise(0.1)
    ])

    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None, augmentation_factor=1):
            '''
            dataset: 원본 데이터셋\
            transform: 증강을 위한 transform
            augmentation_factor: 몇 배로 Augmentation
            '''
            self.dataset = dataset
            self.transform = transform
            self.augmentation_factor = augmentation_factor
            self.original_length = len(dataset)

        def __len__(self):
            # 전체 데이터 수 = 원본 * 배수
            return self.original_length * self.augmentation_factor

        def __getitem__(self, idx):
            # 원본 인덱스를 순환해서 접근
            original_idx = idx % self.original_length
            x, y = self.dataset[original_idx]

            # 증강 적용
            if self.transform:
                x = self.transform(x)

            return x, y
        
    # Augmentation을 적용한 데이터셋 생성
    # 데이터셋을 먼저 train과 val로 나누고, train에 대해서만 증강을 적용
    n_val = int(len(trainset) * 0.2)
    n_train = len(trainset) - n_val
    BATCH_SIZE = 256

    augset, valset = torch.utils.data.random_split(trainset, [n_train, n_val], generator=torch.Generator().manual_seed(2025))

    augset = AugmentedDataset(augset, transform=transform, augmentation_factor=10)

    train_loader = torch.utils.data.DataLoader(
        dataset     = augset, batch_size  = BATCH_SIZE,
        shuffle     = True,num_workers = 0) 

    val_loader = torch.utils.data.DataLoader(
        dataset     = valset, batch_size = BATCH_SIZE,
        shuffle     = False,num_workers = 0)

    # data size check
    print("Train data size:", len(augset),"Val data size:", len(valset),"Test data size:", len(testset))

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

    import torch.nn as nn
    import model

    def get_model_classes():
        """
        model 폴더 내에서 nn.Module 기반 클래스만 자동으로 dict로 반환
        """
        model_classes = {}
        for k in dir(model):
            obj = getattr(model, k)
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj.__module__.startswith('model.'):
                model_classes[k] = obj
        return model_classes

    model_classes = {name: cls() for name, cls in get_model_classes().items()}
    print("Available models:", model_classes.keys())

    # loss function
    from loss.losses import FlexibleLoss
    loss_functions = {
        "MSE": FlexibleLoss(mode="mse"),
        "MSE+SSIM": FlexibleLoss(mode="mse+ssim"),
        "MSE+SSIM+Perceptual": FlexibleLoss(mode="mse+ssim+perceptual"),
        "MSE+Perceptual": FlexibleLoss(mode="mse+perceptual"),
        "SSIM": FlexibleLoss(mode="ssim"),
        "SSIM+Perceptual": FlexibleLoss(mode="ssim+perceptual"),
        "Perceptual": FlexibleLoss(mode="perceptual"),
    }

    EPOCH = 100
    PATIENCE = 20
    BATCH_SIZE = 256
    SAVE_DIR = './checkpoints'
    import os
    import pandas as pd
    from train import MultiModelTrainer
    torch.multiprocessing.set_start_method('spawn', force=True)
    trainer = MultiModelTrainer(
        models=model_classes,     # {"CAE": CAE(), "VAE": VAE(), ...}
        criterions=loss_functions, # {"MSE": ..., "MSE+SSIM": ..., ...}
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=50,
        patience=10,
        save_dir="./checkpoints",
        device=torch.device('cuda'),
        verbose=True,
        max_workers=3,
    )
    trainer.run()
