# 📖 Models Documentation

본 문서는 본 프로젝트에서 사용한 15종의 모델 구조, 특징, 목적 및 참고 문헌을 정리한 자료입니다.

---

## 1️ Autoencoder (AE)
- **모델 특징**: 가장 기본적인 Autoencoder 구조
- **목적**: 입력 이미지 복원 및 재구성 기반 이상탐지
- **주요 function 및 아이디어**: 
    - Encoder-Decoder 구조
    - MSE 기반 Reconstruction Loss
- **Structure**: FC 기반 간단한 Encoder + Decoder
- **참고 문헌**: 
    - Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the Dimensionality of Data with Neural Networks.

---

## 2️ CAE (Convolutional Autoencoder)
- **모델 특징**: CNN 기반 AE
- **목적**: 이미지의 지역적 특징 보존 및 재구성
- **주요 function 및 아이디어**:
    - Convolution + Pooling 기반 Encoder
    - Deconvolution 기반 Decoder
- **Structure**: CNN Encoder → CNN Decoder
- **참고 문헌**:
    - Masci, J. et al. (2011). Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction.

---

## 3️ CVAE (Convolutional Variational Autoencoder)
- **모델 특징**: CNN + VAE 결합
- **목적**: 잠재 공간 분포 학습 + 이미지 재구성
- **주요 function 및 아이디어**:
    - Convolution 기반 Encoder
    - Sampling (Reparameterization Trick)
    - Convolutional Decoder
- **Structure**: Conv Encoder → Latent z → Conv Decoder
- **참고 문헌**:
    - Kingma, D.P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

---

## 4️ DeepCAE
- **모델 특징**: Deep한 CNN 기반 AE
- **목적**: 더 복잡한 Feature Extraction
- **주요 function 및 아이디어**:
    - 깊은 Convolution Layer
- **Structure**: Deep Conv Encoder → Deep Conv Decoder
- **참고 문헌**:
    - Xu, X. et al. (2018). Anomaly Detection Based on Deep Convolutional Autoencoder.

---

## 5️ DeepCVAE
- **모델 특징**: Deep Convolution + VAE
- **목적**: Deep 구조를 통한 효과적인 latent distribution 학습
- **주요 function 및 아이디어**:
    - Deep Conv + Reparameterization
- **Structure**: Deep Conv Encoder → z → Deep Conv Decoder
- **참고 문헌**:
    - Sohn, K. et al. (2015). Learning Structured Output Representation using Deep Conditional Generative Models.

---

## 6️ DeepVAE
- **모델 특징**: Fully Connected 기반 Deep VAE
- **목적**: MNIST 등 low-res dataset에 적합
- **주요 function 및 아이디어**:
    - FC + Reparameterization
- **Structure**: FC Encoder → z → FC Decoder
- **참고 문헌**:
    - Kingma, D.P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

---

## 7️ DenoisingAutoencoder
- **모델 특징**: Noise Robust AE
- **목적**: 노이즈 환경에서도 정상적인 복원
- **주요 function 및 아이디어**:
    - 입력 이미지에 노이즈 추가 후 복원 학습
- **Structure**: AE 구조
- **참고 문헌**:
    - Vincent, P. et al. (2008). Extracting and Composing Robust Features with Denoising Autoencoders.

---

## 8️ DiffusionUNet
- **모델 특징**: UNet 구조 기반 DDPM
- **목적**: Diffusion 기반 Anomaly Detection
- **주요 function 및 아이디어**:
    - UNet + Denoising Score Matching
- **Structure**: UNet
- **참고 문헌**:
    - Ho, J. et al. (2020). Denoising Diffusion Probabilistic Models.

---

## 9️ GANomaly
- **모델 특징**: GAN + AE
- **목적**: Adversarial Training 기반 이상탐지
- **주요 function 및 아이디어**:
    - Generator (AE 역할)
    - Discriminator
    - Adversarial Loss + Reconstruction Loss
- **Structure**: Generator (AE) + Discriminator
- **참고 문헌**:
    - Akçay, S. et al. (2018). GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

---

## 10 HybridCAE
- **모델 특징**: Hybrid CNN 구조 적용 AE
- **목적**: Local + Global feature capture
- **주요 function 및 아이디어**:
    - Multi-Scale Encoder
- **Structure**: CNN 기반 Hybrid Encoder + Decoder
- **참고 문헌**:
    - 자체 설계

---

## 11 RobustAutoencoder
- **모델 특징**: Robust AE
- **목적**: 이상치에 강건한 AE
- **주요 function 및 아이디어**:
    - Robust Principal Component Analysis (RPCA) 아이디어 기반
- **Structure**: FC 기반 Encoder, Decoder
- **참고 문헌**:
    - Zhou, C. et al. (2017). Anomaly Detection with Robust Deep Autoencoders.

---

## 12 SimpleDDPM
- **모델 특징**: Simple DDPM 구조
- **목적**: Lightweight diffusion model
- **주요 function 및 아이디어**:
    - Simplified DDPM Pipeline
- **Structure**: Simple UNet backbone
- **참고 문헌**:
    - Ho, J. et al. (2020). Denoising Diffusion Probabilistic Models.

---

## 13 SkipConnectionAutoencoder
- **모델 특징**: Skip Connection 적용 AE
- **목적**: AE + Residual Learning
- **주요 function 및 아이디어**:
    - Encoder Feature를 Decoder에 Skip 연결
- **Structure**: AE + Skip Connection
- **참고 문헌**:
    - Mao, X. et al. (2016). Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections.

---

## 14 TransformerAnomalyDetector
- **모델 특징**: ViT 기반 Anomaly Detection
- **목적**: Global Context 기반 이상탐지
- **주요 function 및 아이디어**:
    - Vision Transformer Encoder
- **Structure**: Transformer Encoder → Reconstruction
- **참고 문헌**:
    - Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.

---

## 15 VAE
- **모델 특징**: 가장 기본적인 VAE
- **목적**: Latent Representation 기반 이상탐지
- **주요 function 및 아이디어**:
    - Reparameterization Trick
- **Structure**: FC Encoder → z → FC Decoder
- **참고 문헌**:
    - Kingma, D.P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

