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

```python
non_drugs = pd.read_csv("non_drugs.csv")
non_drugs
```

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

이전에 준비한 drug DataFrame에 추출한 feature열을 추가하자

```python
drugs['MW'] = mw
drugs["HBA"] = hba
drugs["HBD"] = hbd
drugs["logp"] = logp
drugs
```

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

이렇게 durg와 non-drug의 SMILES아 특징 값을 가지는 테이블이 완성되었다 <br>

2개의 Table을 합쳐 하나로 만들자

```python
all_data = pd.concat([drugs, non_drugs], ignore_index=True) # drugs, non_drugs 를 합쳐서 all_data라는 DataFrame을 생성
all_data
```

### Support Vector Machine (SVM)

SVM은 Random Forest, NeuralNet 과 더불어 가장 많이 사용되는 ML기법이다. <br>

SVM은 고차원에서 두개의 그룹을 가장 넓게 분리하는 가상의 초평면(hyperplane)을 찾아내는 방법이다. <br>

이때 각 그룹에서 가상의 구분선까지의 거리가 가장 짧은 데이터를 support vector라고 하고, 이때의 간격을 margin이라고 한다. <br>

초평면은 선형일수도 있고, 곡선일수도 있는데 이는 kernel을 바꾸면서 조절할 수 있다.<br>

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

분자의 feature는 3번째 열부터이므로 입력데이터는 3번째열부터 끝까지이다.

```python
X = all_data.iloc[:, 2:] # all_data 에서 전체 행 & 3번째 열~마지막열
X
```

```python
X.describe()
```

목적 값(y)는 is_durg 이름의 열에 해당한다

```python
y = all_data["is_drug"] # all_data 의 is_drug의 열
y
```

#### preprocessing

MCD같은 다른 분석방법도 마찬가지지만, SVM은 결정 경계(decision boundary)를 찾을때 내적(inner product)를 사용하고 거리에 민감한 구조이기때문에 feature마다 단위나 범위가 다르면 스케일이 큰 피처가 영향력을 다 가져가기때문에 정규화를 해줘야한다.

```python
from sklearn import preprocessing
standard_scaler = preprocessing.StandardScaler() # 정규분포로 바꾸어 준다 (표준 점수로 바꾸어 준다.)
X_scaled = standard_scaler.fit_transform(X) # fit_transform이라고 하는 메소드 사용
X_scaled
```

```python
X_scaled_df = pd.DataFrame(X_scaled)
X_scaled_df
```

```python
X_scaled_df.describe()
```

#### Segmentation training / test set

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2) # 20%의 데이터만 테스트용으로 사용.
X_test
```

```python
y_train
```

```python
y_test
```

### Training

```python
my_model.fit(X_train, y_train)
```

```python
y_pred = my_model.predict(X_test)
y_pred
```

## Evaluation

```python
from sklearn.metrics import precision_score, recall_score, f1_score
```

#### Precision

```python
precision_score(y_test, y_pred) # (true, pred)
```

#### Recall

```python
recall_score(y_test, y_pred)
```

#### F1-score

```python
f1_score(y_test, y_pred)
```