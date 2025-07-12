import torch
import torch.nn.functional as F

def fft_loss(x, y):
    """
    FFT 기반 loss: 주파수 도메인에서 magnitude 차이를 기반으로 L1 loss를 계산
    """
    x_fft = torch.fft.fft2(x)
    y_fft = torch.fft.fft2(y)

    # magnitude (진폭) 추출
    x_mag = torch.sqrt(x_fft.real**2 + x_fft.imag**2 + 1e-8)
    y_mag = torch.sqrt(y_fft.real**2 + y_fft.imag**2 + 1e-8)

    # 고주파 영역 차이에 대해 L1 Loss 계산
    loss = F.l1_loss(x_mag, y_mag)

    return loss
