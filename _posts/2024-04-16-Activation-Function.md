---
layout: post
title: Activation Function, What is it?
tags: [ANN, CSE, Data Science, AI]
feature-img: "assets/img/0.post/2024-04-16/header.png"
thumbnail: "assets/img/0.post/2024-04-16/header.png"
categories: CSE, AI
---

&emsp;**활성화 함수**(Activation function)란 **퍼셉트론**(Perceptron)의 출력값을 결정하는 함수이다. 많은 종류의 활성화 함수가 존재하고, 활성화 함수의 결정이 결과값에 크게 영향을 준다. 인공신경망 관련 글은 [**이곳**](https://koderwiki.github.io/cse/2024/04/12/ANN.html) 참조

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/7aed300d-0621-407a-8829-43362d7f0f3a)


## 선형(Linear)과 비선형(Non-Linear)

#### 선형함수 (Linear Function)

**선형함수**(Linear Function)는 말 그래도 직선적인 함수(y=x)이다. 

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/c96eb0e7-d786-4d6b-8d03-ebd4f72e82b0)

하지만 퍼셉트론에서 값을 받고 다른 계층(Layer)로 전달할 때 연속성 있는 **비선형(non-linear)함수**를 사용한다. 비선형 함수를 사용하는 이유는 선형함수를 활성화 함수로 하게 되면 **심층 신경망**(Deep Neural Network)에 큰 도움이 되지 않기 때문이다. 선형 함수를 사용하게 되면, 여러 계층을 통과하는 결과 값을 단 하나의 계층으로 표현할 수 있게 되면서 **신경망을 깊게 쌓는 의미가 사라진다**. 아래의 그림으로 예를 들어보자.

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/78748302-97d5-46a0-8566-f05a319ae4d3)

2개의 계층을 쌓아봤지만, X에 곱해지는 항들은 W로 치환 가능하고, 입력과 무관한 상수들은 전체를 B로 치환 가능하기 때문에 WX + B라는 **동일한 결과**를 낸다.

이처럼 활성화 함수로 선형 함수를 이용하면 여러 층으로 구성하는 이점을 살릴수 없기 때문에 층을 쌓기 위에서는 비선형 함수를 사용해야 한다. 다른 이유로는 XOR gate같은 그래프를 구분할수 없다는 점과 활성화 함수가 비선형일때, 2계층을 가진 신경망은 보편 함수의 근사치(Universal function approximator)임을 증명할수 있다는 점(보편근사정리, Universal approximation theorem) 등이 있다.

## 비선형함수 (Non-Linear Function)

#### 시그모이드 함수 (Sigmoid Fuction)

**시그모이드 함수**(Sigmoid Fuction)은 로짓(Logit)의 역변환이며, **로지스틱 함수**(Logistic Fuction)와 유사한 개념이다.

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/6ed1810e-05e5-403e-8346-24a752b86cfe)

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/ff672403-6823-44d0-90bc-70af771b03a2)

**시그모이드 함수 특징**

> - 시그모이드 함수의 치역은 (0,1), 즉, 0<σ(x)<1
> - 모든 입력값에 대해 출력값이 실숫값으로 정의(Soft Decision)

모든 실수 값을 0보다 크고 1보다 작은 미분 가능한 수로 변환하는 특징을 갖기 때문에, Logistic Classification과 같은 분류 문제에서 자주 사용 된다. 또한 sigmoid()의 return값이 확률 값이기 때문에 결과를 확률로 해석할 때 유용하다.

> - 출력이 0 ~ 1 사이로 확률 표현 가능(Binary Classification)

**시그모이드 함수 한계**

> - 시그모이드 함수는 음수 값을 0에 가깝게 표현하기 때문에 입력 값이 최종 계층(Layer)에서 미치는 영향이 적어지는 **기울기 소실 문제**(Vanishing Gradient Problem)가 발생한다.

시그모이드 도함수 그래프에서 미분 계수를 보면 최대 값이 0.25이다. 딥러닝에서 학습을 위해 **역전파**(Back-propagation)을 계산하는 과정에서 미분 값을 곱하는 과정이 포함되는데, 시그모이드 함수의 경우 은닉층의 깊이가 깊으면 오차율을 계산하기 어렵다는 문제가 발생해서, 기울기 소실 문제가 발생한다. <br>
다시 말해 **x의 절대값이 커질수록 기울기 역전파시 미분 값이 소실 될 가능성이 큰 단점**이 있다.

> - 시그모이드 함수는 중심이 0이 아니라는 점때문에 학습이 느려진다.

#### 하이퍼볼릭 탄젠트(Tanh, Hyperbolic tangent) 함수

**하이퍼볼릭 탄젠트**(tanh, Hyperbolic tangent) 함수는 쌍곡선 함수 중 하나로 다음과 같이 표현된다.

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/0cce376c-8ace-40ed-98b6-a80f44adfcb7)

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/9f274099-719e-4999-b6bc-4b3ceb823b92)

**하이퍼볼릭 탄젠트 함수 특징**

> - 하이퍼볼릭 탄젠트 함수의 치역은 (-1,1), 즉, −1<σ(x)<1
> - 0을 중심으로 함

하이퍼볼릭 탄젠트 함수는 중심점을 0으로 옮겨 시그모이드 함수가 갖오있던 최적화 과정에서 느려지는 문제를 해결했다.

**하이퍼볼릭 탄젠트 함수 한계**

> - **기울기 소실 문제** 발생

#### ReLU(Rectrified Linear Unit) 함수

**ReLU(Rectified Linear Unit, 경사) 함수**는 가장 많이 사용되는 활성화 함수중 하나로, y=x인 **선형 함수**(Linear Fuction)가 **입력값 0 이하에서 부터 정류**(rectified)된 함수다.

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/e54c6241-43ac-4b6e-94d8-234c9b0fa1ba)

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/ed74743f-a04b-468f-928d-230ed8b270a7)

**ReLu 함수 특징**

> - 시그모이드, 하이퍼볼릭탄젠트 함수의 기울기 소실 문제 해결

기울기(미분값)이 0또는 1의 값을 가지기 때문에 시그모이드 함수에서 나타나는 기울기 소실 문제가 발생하지 않는다. 엄연히 비선형(Non-Linear)함수이기 때문에 계층을 깊게 쌓을수 있다.

> - 구현이 단순하고 연산이 필요없어 연산 속도 빠름

exp()함수를 실행하지 않고 임계값(양수/음수 여부)만 활용하기 때문에, 시그모이드 함수나 하이퍼볼릭탄젠트 함수보다 6배정도 빠르게 학습을 할수 있다.

**ReLU 함수 한계**

> - 입력값이 음수일 경우 학습이 안됨(**Dying ReLU**)

입력값이 음수일 경우 출력값과 미분값을 모두 0으로 만들기 때문에 뉴런의 출력물이 0보다 작아서 활성화가 안되며 오차 역전파(Back-propagation)도 전달이 안된다.

#### Leaky ReLU 함수

**Leaky ReLU 함수**는 ReLU 함수가 갖는 **Dying ReLU**현상을 해결하기 위해 나온 변형된 ReLu 함수중 하나이다.

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/8e870fea-c35a-447f-817c-38a01942f07a)

![image](https://github.com/KoderWiki/koderwiki.github.io/assets/153072257/d1bca237-fb78-4d5a-a67f-a47d77b7c161)

**Leaky ReLU 함수 특징**

> - 입력값이 음수일때 출력값을 0이 아닌 매우 작은 값을 출력하여 Dying ReLU현상을 방지함

입력값이 음수일때 0이 아닌 0.01, 0.001과 같은 매우 작은 값을 출력함으로써 입력값이 음수라도 기울기가 0이되는 현상을 방지해 Dying ReLU 현상을 방지한다.

















## Reference
[Activation function | Wikipedia](https://en.wikipedia.org/wiki/Activation_function)<br>
[활성화 함수: 정의와 중류, 비선형 함수를 사용해야 하는 이유](https://kevinitcoding.tistory.com/entry/%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98-%EC%A0%95%EC%9D%98%EC%99%80-%EC%A2%85%EB%A5%98-%EB%B9%84%EC%84%A0%ED%98%95-%ED%95%A8%EC%88%98%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%B4%EC%95%BC-%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0) <br>
[딥러닝 이론, 5. 활성화 함수](https://bbangko.tistory.com/5) <br>
[딥러닝 - 활성화 함수](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=handuelly&logNo=221824080339)<br>
[활성화 함수 개념 및 종류](https://heytech.tistory.com/360)



