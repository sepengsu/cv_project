import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch
import random
def set_seed(seed):
    """
    모든 랜덤 시드 고정 (재현성 확보)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # GPU가 있다면
    torch.backends.cudnn.deterministic = True  # CuDNN deterministic mode
    torch.backends.cudnn.benchmark = False  # CuDNN auto-tuner 비활성화
def weights_init(m):
    """
    CAE / SwinAE 프로젝트에 적절한 초기화:
    - Conv 계열: Kaiming Normal (ReLU용)
    - Linear 계열: Xavier Uniform
    - Bias는 모두 0
    """
    set_seed(2025)  # 랜덤 시드 고정 (재현성 확보)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
