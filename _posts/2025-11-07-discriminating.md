---
date: 2025-11-06 10:03:00
layout: post
title: Discriminating drgs and non-drugs using SVM
subtitle: 'SMILES classification'
description: >-
  SVM을 이용해 약물과 비약물을 분류하자
image: >-
  https://koderwiki.github.io/assets/img/0.post/linear/drugsvm.png
optimized_image: >-
  https://koderwiki.github.io/assets/img/0.post/linear/drugsvm.png
category: Machine Learning
tags:
  - SVM
  - Machine Learning
  - Classification
  - CADD
  - Data Science
  - blog
author: geonu.Ko
paginate: true
use_math : true
---


## Discriminating drugs and non-drugs using SVM

### Introduction

Alpha-fold, RoseTTAFold 등 다양한 인공지능 기반 제약 연구의 급속한 발전속에서, 잠재적인 약물 후보와 비약물 화합물을 포함한 다양한 화합물 데이터베이스가 구축되고 있다. <br>

이러한 데이터 속에서 약물(drug)과 비약물(non-drug)을 효율적으로 분류하는 것은 컴퓨터 기반 신약 개발(Computer-Aided Drug Discovery, CADD) 및 가상 스크리닝 (Virtual Screening) 과정에서 매우 중요한 단계이다. <br>

서포트 벡터 머신(Support Vector Machine, SVM) 은 다양한 분류(Classification) 문제에서 강력하고 신뢰할 수 있는 지도학습 기법으로 널리 활용된다. <br>

본 글에서는 SVM을 활용해서 SMILES(Simplified Moldecular Input Line Entry System)를 기반으로 약물과 비약물을 구분하고 가장 잘 분류하는 parameter 를 찾아보자. <br>

### Import Necessary Libraries

```python
import sklearn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```

```python
import rdkit
import rdkit.Chem as Chem
```

### Loading Data

```python
drugs = pd.read_csv("drugs.csv")
drugs
```
<img width="822" height="562" alt="image" src="https://github.com/user-attachments/assets/90c4531c-20e5-472d-8c4a-802d89007d4d" />

```python
non_drugs = pd.read_csv("non_drugs.csv")
non_drugs
```
<img width="813" height="550" alt="image" src="https://github.com/user-attachments/assets/92bb0312-7e80-4025-a3d4-9d53084d34ae" />



### Molecular Descriptor Calculation (Feature Extraction)

```python
# 분자들의 특징 (descriptor) 을 뽑아내자. 
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcCrippenDescriptors, CalcNumLipinskiHBA, CalcNumLipinskiHBD

# 빈 리스트 준비. 
mw = []   # molecular weight
hba = []  # hbond acceptor
hbd = []  # hbond donor
logp = [] # logP 

# 분자를 차례로 반복하면서 특징 값을 계산. 
for smi in drugs["smiles"]:
    m = Chem.MolFromSmiles(smi)
    mw.append(CalcExactMolWt(m))
    logp.append(CalcCrippenDescriptors(m)[0]) # because calccrippendescriptors returns two values: logp, mr
    hba.append(CalcNumLipinskiHBA(m))
    hbd.append(CalcNumLipinskiHBD(m))
```

```python
mw
```
<img width="255" height="900" alt="image" src="https://github.com/user-attachments/assets/07ef6489-fd79-4dfa-ba95-d0fc2662d166" />


이전에 준비한 drug DataFrame에 추출한 feature열을 추가하자

```python
drugs['MW'] = mw
drugs["HBA"] = hba
drugs["HBD"] = hbd
drugs["logp"] = logp
drugs
```

<img width="1109" height="500" alt="image" src="https://github.com/user-attachments/assets/9de02551-e7d6-4c35-be63-675b1ceb84ab" />



<br>

non-drug에도 위 과정을 반복하자

```python
mw = []
hba = []
hbd = []
logp = []
 
for smi in non_drugs["smiles"]:
    m = Chem.MolFromSmiles(smi)
    mw.append(CalcExactMolWt(m))
    logp.append(CalcCrippenDescriptors(m)[0]) # because calccrippendescriptors returns two values: logp, mr
    hba.append(CalcNumLipinskiHBA(m))
    hbd.append(CalcNumLipinskiHBD(m))
```

```python
non_drugs["MW"] = mw
non_drugs["HBA"] = hba
non_drugs["HBD"] = hbd
non_drugs["logp"] = logp
non_drugs
```
<img width="1134" height="495" alt="image" src="https://github.com/user-attachments/assets/2be8378e-5b02-4291-b914-499f7c119f93" />


이렇게 durg와 non-drug의 SMILES아 특징 값을 가지는 테이블이 완성되었다 <br>

2개의 Table을 합쳐 하나로 만들자

```python
all_data = pd.concat([drugs, non_drugs], ignore_index=True) # drugs, non_drugs 를 합쳐서 all_data라는 DataFrame을 생성
all_data
```
<img width="1163" height="544" alt="image" src="https://github.com/user-attachments/assets/3c40e341-0615-4e95-82ad-8a720df1b4e0" />


### Support Vector Machine (SVM)

SVM은 Random Forest, NeuralNet 과 더불어 가장 많이 사용되는 ML기법이다. <br>

SVM은 고차원에서 두개의 그룹을 가장 넓게 분리하는 가상의 초평면(hyperplane)을 찾아내는 방법이다. <br>

이때 각 그룹에서 가상의 구분선까지의 거리가 가장 짧은 데이터를 support vector라고 하고, 이때의 간격을 margin이라고 한다. <br>

초평면은 선형일수도 있고, 곡선일수도 있는데 이는 kernel을 바꾸면서 조절할 수 있다.<br>

<img width="943" height="904" alt="image" src="https://github.com/user-attachments/assets/9923a032-ec21-4091-8680-220a3b037768" />



### Discriminating durgs and non-drugs using SVM

scikit-learn에서 svm 모듈을 불러오자.

```python
from sklearn import svm 
# support vector classifier
my_model = svm.SVC()
```

전체 데이터를 학습용과 테스트용으로 분리하자

```python
from sklearn.model_selection import train_test_split # train_test_split 함수를 불러오자
all_data 
```
<img width="1160" height="497" alt="image" src="https://github.com/user-attachments/assets/ae6ed231-09d4-430b-b621-bc6c88255754" />


분자의 feature는 3번째 열부터이므로 입력데이터는 3번째열부터 끝까지이다.

```python
X = all_data.iloc[:, 2:] # all_data 에서 전체 행 & 3번째 열~마지막열
X
```
<img width="466" height="504" alt="image" src="https://github.com/user-attachments/assets/e860b10c-7bfd-43b6-98a8-61620c5027d6" />


```python
X.describe()
```
<img width="640" height="374" alt="image" src="https://github.com/user-attachments/assets/ae8fab7c-bf2f-40cd-8b8d-616c23956951" />


목적 값(y)는 is_durg 이름의 열에 해당한다

```python
y = all_data["is_drug"] # all_data 의 is_drug의 열
y
```
<img width="499" height="354" alt="image" src="https://github.com/user-attachments/assets/b603652c-6566-4e63-b107-69c3d0ebef04" />


#### preprocessing

MCD같은 다른 분석방법도 마찬가지지만, SVM은 결정 경계(decision boundary)를 찾을때 내적(inner product)를 사용하고 거리에 민감한 구조이기때문에 feature마다 단위나 범위가 다르면 스케일이 큰 피처가 영향력을 다 가져가기때문에 정규화를 해줘야한다.

```python
from sklearn import preprocessing
standard_scaler = preprocessing.StandardScaler() # 정규분포로 바꾸어 준다 (표준 점수로 바꾸어 준다.)
X_scaled = standard_scaler.fit_transform(X) # fit_transform이라고 하는 메소드 사용
X_scaled
```
<img width="724" height="242" alt="image" src="https://github.com/user-attachments/assets/13fc5d74-6c8b-4712-8a1f-102855f8ab77" />


```python
X_scaled_df = pd.DataFrame(X_scaled)
X_scaled_df
```
<img width="559" height="546" alt="image" src="https://github.com/user-attachments/assets/e96d3d89-8cc3-4221-b374-cd335a42b7c0" />


```python
X_scaled_df.describe()
```
<img width="736" height="379" alt="image" src="https://github.com/user-attachments/assets/e11f57de-472b-4dfe-abc5-f5a1c1aa2b76" />


#### Segmentation training / test set

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2) # 20%의 데이터만 테스트용으로 사용.
X_train
```
<img width="559" height="543" alt="image" src="https://github.com/user-attachments/assets/a33f79a8-a626-4183-9fc6-de5669802825" />


```python
X_test
```
<img width="560" height="554" alt="image" src="https://github.com/user-attachments/assets/21036b00-46f2-4179-8ecf-206b4862d598" />


```python
y_train
```
<img width="523" height="357" alt="image" src="https://github.com/user-attachments/assets/8cf7f5c2-11fe-44c7-9d65-3d469f730f0a" />


```python
y_test
```
<img width="523" height="357" alt="image" src="https://github.com/user-attachments/assets/57e7ccb8-04ed-47d3-ad22-45a5e65c78ec" />


### Training

```python
my_model.fit(X_train, y_train)
```

```python
y_pred = my_model.predict(X_test)
y_pred
```
<img width="862" height="448" alt="image" src="https://github.com/user-attachments/assets/c89b7fc1-7480-47a0-aeb4-033c819b00a5" />


## Evaluation

```python
from sklearn.metrics import precision_score, recall_score, f1_score
```

대표적인 기법인 Precision, Recall, F1-score로 평가하자. <br>

#### Precision

<img width="556" height="360" alt="image" src="https://github.com/user-attachments/assets/4290d459-9f80-4685-89fd-b0e3f1d5ad65" />


Precision은 분류 모델이 Positive로 판정한 것 중, 실제로 Positive인 샘플의 비율이다. <br>

Precision은 Positive로 검출된 결과가 얼마나 정확한지를 나타낸다.

$$
Precision = \frac {TP}{TP + FP}
$$

```python
precision_score(y_test, y_pred) # (true, pred)
```
<img width="395" height="55" alt="image" src="https://github.com/user-attachments/assets/babe3296-aa48-4042-b84a-14bdc3ccab99" />


약 53%의 정확도가 나왔다. 즉, 약이라고 예측 한 것 중에 53%가 맞았다.

#### Recall

<img width="553" height="379" alt="image" src="https://github.com/user-attachments/assets/66cf21a5-34f0-4f1e-8bc4-3689ddc126a3" />


Recall은 실제 Positive 샘플 중 분류 모델이 Positive로 판정한 비율이다. <br>

통계학에서는 Sensitivity라고도 한다. Recall은 분류 모델이 실제 Positive 클래스를 얼마나 빠지지 않고 잘 잡아냈는지를 나타낸다.

$$
Recall = \frac {TP}{TP+FN}
$$

```python
recall_score(y_test, y_pred)
```
<img width="395" height="48" alt="image" src="https://github.com/user-attachments/assets/787fb7b3-52e3-44b1-9033-e0e16b65578d" />

실제 약 중에서 14%만 제대로 찾았다. 상당히 낮은 수치이다.

#### F1-score

분류 모델에서 Precisiton과 Recall 성능을 동시에 고려하기 위해서 F1-score라는 지표를 사용할 수 있다. F1-score는 Precision과 Recall의 조화평균으로 정의된다. <br>

이는 0에서 1사이값이며 1에 가까울 수록 성능이 좋음을 나타낸다.

$$
F1-score = 2 \ \text{x} \ \frac {Precision\ \text{x} \ Recall}{Precision + Recall}
$$

```python
f1_score(y_test, y_pred)
```
<img width="433" height="62" alt="image" src="https://github.com/user-attachments/assets/7d28d544-6cc7-4f40-a6ee-c9eb0cd0a473" />


## SVC class in scikit-learn

SVC에는 성능을 향상시키기위한 많은 parameter가 존재한다. <br>

### regularization parameter : C

soft-margin svm에서 나오는 parameter C이다. <br>

C값이 작으면, 다른 그룸에서 속하는 데이터 사이의 간격을 최대한 넓히도록 학습이 되고, 어느정도 오분류는 허용한다. <br>

C값이 크면 다른 그룹에 속하는 데이터 사이의 간격이 좁아지는 것을 허용하지만, 분류 오류를 최소화 하는 방향으로 학습이 된다. <br>

<img width="1040" height="558" alt="image" src="https://github.com/user-attachments/assets/1085987a-d970-448f-b9d0-9ab45f3c542e" />


C=10으로 테스트해보자

```python
my_model_v2 = svm.SVC(C=10) # default 값은 C=1인데, 이 모델에서는 C=10을 사용해서 테스트!
```

```python
my_model_v2.fit(X_train, y_train)
```

```python
y_pred_v2 = my_model_v2.predict(X_test)
y_pred_v2
```
<img width="895" height="282" alt="image" src="https://github.com/user-attachments/assets/e51bf300-eaf4-4ea8-ad4d-af653dd52815" />


```python
f1_score(y_test, y_pred_v2)
```
<img width="394" height="66" alt="image" src="https://github.com/user-attachments/assets/a3140134-2aa1-43b9-8543-964f43ec94a7" />


아까보다 약간 성능이 향상된 것을 볼 수 있다.

```python
precision_score(y_test, y_pred_v2)
```
<img width="412" height="67" alt="image" src="https://github.com/user-attachments/assets/497e23dc-fe83-4d35-8414-a3c3cb497c4c" />


```python
recall_score(y_test, y_pred_v2)
```
<img width="438" height="45" alt="image" src="https://github.com/user-attachments/assets/4104e38f-91e3-41be-9b66-6518d0dd09fe" />


정확도는 비슷하지만 recall이 향상되었다.<br>

### gamma

gamma 값은 데이터 값이 어느정도로 주변에 영향을 미치는지를 결정한다. <br>

<img width="942" height="712" alt="image" src="https://github.com/user-attachments/assets/5012290b-ccc6-4126-95d6-024fa3412df5" />


gamma 값이 크면, 각 데이터 값은 그 주변에만 영향을 미치고 곡선이 복잡해지는 반면, <br>

gamma값이 작으면, 넓은 영역에 영향을 미쳐 직선에 가까워진다. <br>


```python
my_model5 = svm.SVC(gamma = 10.0)
```

```python
my_model5.fit(X_train, y_train)
y_pred = my_model5.predict(X_test)
```

```python
f1_score(y_test, y_pred)
```
<img width="408" height="67" alt="image" src="https://github.com/user-attachments/assets/500a90cd-6427-4727-b7da-07962220d325" />


```python
precision_score(y_test, y_pred)
```
<img width="306" height="57" alt="image" src="https://github.com/user-attachments/assets/8d171353-c1a4-488e-9ad2-6b81aa9a2177" />


```python
recall_score(y_test, y_pred)
```
<img width="451" height="61" alt="image" src="https://github.com/user-attachments/assets/aa018e05-df5e-4225-9ded-c16ae8b21822" />


<br>

### Kernnel

SVM은 다양한 커널이 존재한다. <br>

> - kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} <br>
>   
> - kernel ~ core <br>
>   
> - 모델의 핵심에 들어있는 수학함수의 형태. <br>
>   
> - linear: 선형 (ax+b) <br>
>   
> - poly: ax^3 + bx^2 + cx + d <br>
>   
> - **rbf**: Gaussian <br>
>   
> - sigmoid: 1/(1+e^x) <br>
>   

```python
my_model_linear = svm.SVC(kernel='linear')
my_model_linear.fit(X_train, y_train)
y_pred_linear = my_model_linear.predict(X_test)
```

```python
f1_score(y_test, y_pred_linear)
```

```python
my_model_poly = svm.SVC(kernel='poly')
my_model_poly.fit(X_train, y_train)
y_pred_poly = my_model_linear.predict(X_test)
f1_score(y_test, y_pred_poly)
```

잘 작동이안된다

## Finding the optimal parameters systematically

반복문으로 최적의 파라미터 조합을 찾아보자

```python
max_f1 = 0.0 
for c in [0.1, 1, 2, 5, 10, 100, 400, 500, 600, 1000]: # test할 다양한 C 값
    for g in [0.001, 0.01, 0.1, 0.5, 1.0, 2, 10, 50, 100]:
        model = svm.SVC(kernel = 'rbf', C=c,gamma=g)
        model.fit(X_train, y_train) # 학습
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        if f1 > max_f1: # 기존의 최고의 F1 값을 넘었을 때. 
            max_f1 = f1
            max_prec = prec
            max_recall = recall
            max_c = c
            max_g = g
            
        print(f"C: {c}\tgamma: {g}\tF1: {f1}\tPrec: {prec}\tRecall: {recall}")
print("--End of Calculation!--")
print(f"max_C: {max_c}\tmax_gamma: {max_g}\tF1: {max_f1}\tPrec: {max_prec}\tRecall: {max_recall}")
```

<img width="1312" height="617" alt="image" src="https://github.com/user-attachments/assets/4b22cafc-c5f1-4886-9e47-222208527a23" />


C = 600, gamma = 2일때 F1-score = 0.46으로 가장 좋았다 <br>

72개의 조합 중 좋은거지 최적의 값은 아니다!


## Reference
[이주용교수,"하이퍼파라미터최적화"](https://www.youtube.com/watch?v=WMstoG_fK-0&t=2058s) <br>
[분류성능지표](https://ai-com.tistory.com/entry/ML-%EB%B6%84%EB%A5%98-%EC%84%B1%EB%8A%A5-%EC%A7%80%ED%91%9C-Precision%EC%A0%95%EB%B0%80%EB%8F%84-Recall%EC%9E%AC%ED%98%84%EC%9C%A8)
