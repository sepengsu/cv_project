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

