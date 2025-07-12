# TEAM 8 Project (2025 Computer Vision)

---

## ✅ 프로젝트 환경 및 주요 라이브러리

### Python 및 주요 라이브러리 버전
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch 2.6.0+cu126](https://img.shields.io/badge/pytorch-2.6.0%2Bcu126-orange.svg)
![NumPy 2.1.2](https://img.shields.io/badge/numpy-2.1.2-blue.svg)
![Pandas 2.2.3](https://img.shields.io/badge/pandas-2.2.3-green.svg)
![scikit-learn 1.6.1](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)
![TensorBoard 2.19.0](https://img.shields.io/badge/tensorboard-2.19.0-purple.svg)
![Matplotlib 3.10.1](https://img.shields.io/badge/matplotlib-3.10.1-red.svg)
![Seaborn 0.13.2](https://img.shields.io/badge/seaborn-0.13.2-blue.svg)
![SciPy 1.15.2](https://img.shields.io/badge/scipy-1.15.2-green.svg)
![Jupyter 6.29.5](https://img.shields.io/badge/jupyter-6.29.5-orange.svg)
![IPython 9.0.2](https://img.shields.io/badge/ipython-9.0.2-yellow.svg)
![TorchSummary 1.5.1](https://img.shields.io/badge/torchsummary-1.5.1-lightgrey.svg)

---

## ✅ 실험 환경
- **GPU**: NVIDIA RTX 4060 Ti 16GB
- **CUDA**: 12.6
- **PyTorch**: 2.6.0+cu126
- **Batch Size**: 1024
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR
- **Mixed Precision**: FP16 Training (Validation은 FP32)
- **Epoch**: Baseline 180 Epoch + EarlyStopping, Fine-Tuning 200 Epoch 예정
- **Data Augmentation**: Fashion MNIST 5~10배 증폭

---

## ✅ 프로젝트 개요
본 프로젝트는 Fashion MNIST 기반의 **Anomaly Detection**을 목표로 하며, 다양한 Reconstruction 및 Diffusion 기반 모델을 비교 실험합니다.

- **모델 수**: 총 15종
- **Loss Function**: Custom Flexible Loss Framework 지원
- **Training 방식**: Grid Search 기반의 대규모 실험
- **Evaluation**: ROC-AUC, PR-AUC, F1-Score, ACC
- **Logging**: TensorBoard 기반 GPU Memory / Loss / Sample Image Logging

---

## ✅ 사용된 주요 기술 스택
- Flexible Loss 조합 (MSE, Charbonnier, SSIM, Gradient, TV 등)
- Autoencoder 기반 모델군
- Diffusion 기반 모델군
- GANomaly 기반 모델
- Vision Transformer 기반 모델
- Grid Search 기반 Loss-Model 조합 실험 자동화
- FP16 Mixed Precision 학습

---

## ✅ 적용된 모델 목록 (총 15종)

| No | 모델명 | 유형 | 특징 |
|----|--------|------|-------|
| 1 | Autoencoder | AE | 기본적인 Autoencoder |
| 2 | CAE | AE | Convolutional Autoencoder |
| 3 | CVAE | VAE | Convolutional Variational Autoencoder |
| 4 | DeepCAE | AE | Deep Convolutional Autoencoder |
| 5 | DeepCVAE | VAE | Deep Convolutional Variational Autoencoder |
| 6 | DeepVAE | VAE | Fully Connected 기반 Deep Variational Autoencoder |
| 7 | DenoisingAutoencoder | AE | Noise Robust Denoising Autoencoder |
| 8 | DiffusionUNet | Diffusion | U-Net 기반 Diffusion 모델 |
| 9 | GANomaly | GAN | GAN 기반 Anomaly Detection (Adversarial + Reconstruction) |
| 10 | HybridCAE | AE | Hybrid 구조를 적용한 Convolutional Autoencoder |
| 11 | RobustAutoencoder | AE | Adversarial Training 기반 Robust Autoencoder |
| 12 | SimpleDDPM | Diffusion | Simplified Denoising Diffusion Probabilistic Model |
| 13 | SkipConnectionAutoencoder | AE | Skip-connection 적용된 Autoencoder |
| 14 | TransformerAnomalyDetector | Transformer | Vision Transformer 기반 Anomaly Detection |
| 15 | VAE | VAE | 기본적인 Variational Autoencoder |

---

> 📚 **각 모델별 상세 설명**  
👉 [모델 설명 문서 바로가기](./docs/models.md)


# 🧠 FFT 기반 Multi-Memory AutoEncoder Summary

## 📌 프로젝트 개요
이 프로젝트는 **Fashion-MNIST 기반의 이상 탐지**를 위해 설계된 복합형 오토인코더(AutoEncoder) 모델이다.
특히, **중앙 질감 중심 복원**과 **고주파 특성(FFT) 기반 이상 검출**, **Memory Matching 기반 인코딩**을 통해 정밀한 분리와 복원을 목표로 한다.

---

## 🏗️ 전체 구조

```mermaid
graph TD;
    x[Input Image (1x28x28)] --> Center[Masked Center Encoder];
    x --> Structure[Structure Encoder];
    x --> FFT[FFT Encoder];

    Center --> MEM1[Center Memory (KQV)] --> zc[Matched Center Feature];
    Structure --> MEM2[Structure Memory (Hard)] --> zs[Matched Structure Feature];
    FFT --> MEM3[FFT Memory (KQV)] --> zf[Matched FFT Feature];

    zc --> Dec[Triple Decoder];
    zs --> Dec;
    zf --> Dec;

    Dec --> x_hat[Reconstructed Image (1x28x28)];
```

### ✨ 구성 요소
- `MaskedCenterEncoder`: Gaussian 중심 마스크를 기반으로 중앙 질감을 추출
- `StructureEncoder`: 전체 구조 정보를 압축
- `FFTEncoder`: 주파수 특성을 추출하는 전용 인코더
- `Memory Modules`:
  - `KeyValueMemory`: center/fft → soft matching (KQV 방식)
  - `HardMemory`: structure → 가장 유사한 하나를 선택 (hard matching)
- `TripleFeatureDecoder`: 세 인코딩 결과를 concat하여 최종 복원


---

## 🎯 Loss Function

```python
loss = FlexibleLoss(
    mode="mse+center_crop+ms-ssim+center_weighted",
    loss_weights={
        "mse": 0.6,
        "center_crop": 1.0,
        "ms-ssim": 0.5,
        "center_weighted": 0.5
    },
    reduction="mean",
    epoch=100
)
```

### ✅ 핵심 구성
| Loss | 설명 |
|------|------|
| `MSE` | 전체 복원 손실 |
| `Center Crop Loss` | 중심 영역만 비교하는 손실 |
| `MS-SSIM` | 질감, 구조 보존 평가 |
| `Center Weighted Loss` | 중심을 강조하는 가중 손실 (Gaussian 기반) |

> 🔍 이 조합은 "전체 복원 + 중심 강조 + 질감 유지"의 균형을 추구함.


---

## 🧠 Memory 초기화 전략

```python
init_memory(model, train_loader, device=device, mem_size=800, top_n=2400, mode='kmeans')
```

### 💾 Key 설정 전략
| 모듈 | Key | Value | 방식 |
|-------|------|--------|-------|
| Center | Frozen | Trainable | KMeans 또는 Mean |
| Structure | N/A | N/A | Hard Match (mean 기반) |
| FFT | Frozen | Trainable | KMeans 또는 Mean |

---

## 📚 참고 문헌 및 영감

- **MemAE: Memory-Augmented Autoencoder for Unsupervised Anomaly Detection**  
  Gong et al., ICCV 2019 [[paper]](https://arxiv.org/abs/1904.02639)
- **SSIM/MS-SSIM Loss for Image Quality** [[source]](https://ece.uwaterloo.ca/~z70wang/research/ssim/)
- **FFT-based Anomaly Detection** 논문 및 최근 diffusion 기반 이상 탐지 문헌
- 자체 아이디어 조합:
  - 중앙 질감 중심 AE
  - FFT → Memory Matching 결합
  - Loss 조합 기반 가중 손실 튜닝

---

## ✅ 향후 계획
- [ ] FFT 특성 보강을 위한 주파수 attention decoder 개발
- [ ] Decoder 중심 중심 masking attention 추가 실험
- [ ] Gaussian 대신 adaptive 중심 강조 방법 도입
- [ ] skip-connection에 대한 질감 matching 강화

---

> 📍 구현: PyTorch 기반, GPU 학습 환경 기준.
> 📅 작성일: 2024-04


