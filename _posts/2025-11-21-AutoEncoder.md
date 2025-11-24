---
date: 2025-11-21 01:35:59
layout: post
title: Auto-Encoder
subtitle: 'Latent Space using Autoencoder'
description: >-
  Autoencoder 학습 과정과 차원 축소
image: >-
  https://koderwiki.github.io/assets/img/0.post/linear/autoencoder.png
optimized_image: >-
  https://koderwiki.github.io/assets/img/0.post/linear/autoencoder.png
category: Computer Vision
tags:
 - Autoencoder
  - Machine Learning
  - Deep Learning
  - Anomaly Detection
  - Data Science
  - blog
author: geonu.Ko
paginate: true
use_math : true
---

## Auto-Encoder

<img width="948" height="482" alt="image" src="https://github.com/user-attachments/assets/74cce6e0-6420-4ff7-8b90-0fc6c0b54c17" />


### Introduction

Auto-Encoder(AE)는 입력데이터를 스스로 압축(Encoding)하고 다시 복원(Decoding)하는 비지도 학습(unsupervised learning) 기반의 딥러닝 구조이다. <br>

주로 feature extraction, 차원 축소 등 다양한 분야에서 사용된다. 기본적으로 reconstruction 기반의 학습을하고 latent space를 추출해 내는 것이 AE의 핵심이다. <br>

loss function과 encoder, decoder의 구조의 따라 VAE, ConvAE 다양한 종류의 AE를 설계할 수 있다. <br>

### Auto-Encoder

<img width="856" height="395" alt="image" src="https://github.com/user-attachments/assets/19104806-a6ce-4c9b-b167-fef3948c4bb8" />


AE는 입력 값 x를 받아, 잠재공간(latent spce)라 불리는 더 작은 차원 z로 변환 한뒤, 원본과 대조하여 유사한 출력 x'을 재구성하는 모델이다. <br>

AE는 입력 데이터의 특징을 추출하고 잠재 벡터 z로 변환하는 **Encoder**

$$
z = f_{enc}(x)
$$

압축된 저차원 벡터인 **Latent Space** <br>

그리고 이를 다시 원본으로 recon하는 **Decoder** 구조로 되어 있다.

$$
\hat{x} = f_{dec}(z)
$$

따라서 전체적인 흐름은 다음과 같다.

$$
x \rightarrow Encoder \rightarrow z(latent \ space) \rightarrow Decoder \rightarrow \hat{x} 
$$

### Encoder

Encoder는 AE에서 전반부를 구성하는 구조로, 입력데이터의 고차원 정보를 더 작은 차원의 latent space로 압축하는 역할을 담당한다. <br>

딥러닝 구조이기 여러개의 layer로 구성되어있으며, layer 구조에 따라 dense layer, convolution layer, variational layer 등으로 설계할 수 있다. <br>

이미지 데이터에 주로 사용하는 2차원 convolution layer인 Conv2D를 예시로 들면 <br>

Con2D : 공간적 특징 저장<br>

Stride 2 / MaxPooling : 다운샘플링 <br>

ReLU : 비선형성 추가 <br>

위 구조가 연결되어 레이어를 거칠수록 점점 압축이 된다 <br>

### Decoder

Decoder는 AE 후반부를 구성하는 요소로, latent space 표현 z를 다시 원래 데이터로 reconstruction하는 역할을 수행한다. 즉, 압축된 정보 기반으로 원본 데이터에 최대한 근접한 출력을 생성하는 역할을 한다. <br>

인코더랑 대칭구조를 이루며 구조도 대칭적이다. <br>

ConvTranspose2D : 해상도 확장 <br>

Stride 2 / MaxPooling : 업 샘플링 <br>

ReLU : 비선형성 추가 <br>

Sigmoid : 최종 출력 범위 조정 <br>

차이점은 마지막 출력은 확률이 필요하기 때문에 활성화 함수를 sigmoid/ tanh를 넣어서 값을 구한다 (ReLU는 0,무한대 발산 구조이기 때문이다) <br>

### Loss function

Loss fucntion(손실함수)는 AE의 출력이 원본과 얼마나 유사한지 정량적으로 측정하는 함수로, 이 값을 최소화 시키는 것이 AE의 학습 목표이다. <br>

손실함수 종류에 따라 학습 결과가 다르게 나오며, Encoder와 Decoder의 전체 학습을 이끈다. <br>

손실함수는 MSE, MAE, SSIM KL Divergence 등 다양하게 있으며, 데이터 형태, 목적에 따라 올바른 함수를 고르는 것이 필수적이다. <br>

MSE와 MAE가 대표적이다.

#### Mean Squared Error (MSE)

$$
L_{MSE} = \frac{1}{n}\sum(y - \hat{y})^2
$$

#### Mean Absolute Error (MAE)

$$
L_{MAE} = \frac{1}{n}\sum |{x - \hat{x}}|
$$

### Optimization

모든 딥러닝에서 마찬가지지만 이 손실함수를 최적화하는 것이 중요하다. <br>

대표적으로 Gradient Descent가 있고 그 외에 SGD, Momentum, 그리고 가장 많이 쓰는 Adam(Adaptive Moment Estimation) 등이 존재한다. <br>

Adam은 1st moment(평균), 2st moment(분산, 제곱평균)을 자동으로 추정해서 학습하고 안정화 시킨다. <br>


### Latent Space

AE의 핵심은 latent space이다. 이 잠재된 축소공간이 AE가 학습한 데이터의 핵심 표현(embedding)으로 이 공간에서 원본 데이터의 중요한 feature만 압축되어 이상치 탐지, 특징 추출, 노이즈 제거 등 다양한 용도로 사용된다.

<img width="522" height="277" alt="image" src="https://github.com/user-attachments/assets/55ae8c9a-0620-4dc3-a414-a7eba8a46b18" />
