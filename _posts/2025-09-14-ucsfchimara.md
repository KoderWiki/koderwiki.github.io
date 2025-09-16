---
date: 2025-09-14 19:24:23
layout: post
title: Protein homology modeling using UCSF Chimera
subtitle: 'UCSF Chimera를 이용한 상동성 모델링'
description: >-
  서열 유사성을 기반으로한 단백질의 3차원 구조 추정
image: >-
  https://koderwiki.github.io/assets/img/0.post/bioimage/cimage.png
optimized_image: >-
  https://koderwiki.github.io/assets/img/0.post/bioimage/cimage.png
category: Protein Modeling
tags:
  - Computer Vision
  - Protein Modeling
  - UCSF Chimera
  - UniProt
  - Ai Based Drug Design
  - blog
author: geonu.Ko
paginate: true
use_math : true
---

## Protein homology modeling using UCSF Chimera

### Introduction

    단백질의 3차원 구조는 그 기능과 생물학적 역할을 결정짓는 핵심 요소이기 때문에, 단백질 구조 예측은 구조생물학과 신약개발 분야에서 매우 중요한 과제이다.

    하지만 모든 단백질에 대한 구조를 알기에는 어려우며, 이러한 한계를 보완하는 접근법이 바로 **상동성 모델링(homology modeling)** 으로, 서열 유사성을 기반으로 알려진 구조(template)를 활용하여 목표 단백질의 3차원 구조를 추정한다.

    여러 소프트웨어 도구들 중 **UCSF Chimera**는 직관적인 인터페이스를 제공하며, 서열 정렬, 상동성 모델 생성, 그리고 분자 시각화를 통합적으로 지원한다. 특히 BLAST나 PSI-BLAST를 이용한 서열 검색과 단백질 데이터뱅크(PDB)의 구조 템플릿을 결합하여 초기 3차원 모델을 생성하고, 이를 시각적으로 분석 및 수정할 수 있다. 이러한 과정은 **단백질–리간드 상호작용 탐색**과 같은 과정에서 중요한 정보들을 제공한다.

**UCSF Chimera**를 이용하여 **서열 유사성**을 기반으로 한 **Protein homology modeling**을 실습해보자.<br>

<br>

### Target Protein

Name : **Gag-Pol polyprotein** <br>

UniProt : **P03366** <br>

Amino acids : 1447 <br>

<br>

Gag-Pol은 HIV에서 구조 단백질과 핵심 효소들을 한꺼번에 담고 있는 Polyprotein이다. <br>

이 단백질은 번역 과정에서 리보솜 프레임시프트를 통해 제한적으로 생성되며, Gag과 Gag-Pol의 합성 비율은 바이러스의 조립과 감염성 유지에 결정적인 역할을 한다. <br>

따라서 Gag-Pol은 HIV 생존과 복제에 필수적일 뿐만 아니라, 항레트로바이러스 약물이 직접적으로 겨냥하는 주요 타깃으로서 임상적으로도 매우 중요한 단백질이다. <br>

<br>

### UniProt

[**UniProt**](https://www.uniprot.org/uniprotkb/P03366/entry#structure) 은 단백질 서열과 기능 정보에 관한 가장 포괄적이고 무료로 이용할 수 있는 생물학 데이터베이스로, 연구 문헌에서 얻은 단백질의 생물학적 기능에 대한 많은 정보를 제공하고 있다. <br>

현재 목표하는 단백질인 Gag-Pol polyprotein 또한 UniProt에서 Structure와 Sequence 등 다양한 정보를 확인할 수 있다. <br>
<br>
**Structure** <br>
<img width="966" height="611" alt="image" src="https://github.com/user-attachments/assets/a6dbb7b6-e377-467f-aab9-9e65cfebe7f9" />
<br>

**Sequence 중 일부** <br>
<img width="1910" height="835" alt="image" src="https://github.com/user-attachments/assets/1e6a3461-1c3d-4a14-a22f-ad132de37f30" />



<br>

### USCF Chimera

**UCSF Chimera** 는 단백질·핵산 등 생체분자의 **3차원 구조를 시각화하고 분석할 수 있는 오픈소스 소프트웨어**이다. <br>구조 모델링, 분자 간 상호작용 탐색, docking 결과 확인 등 다양한 기능을 지원하며, 연구자들이 **단백질 구조 예측과 시각화**에 널리 활용한다. <br>

이제 본격적으로 USCF Chimera를 이용해 Modeling 해보자. <br>

<br>

### Modeling

download : [UCSF Chimera Home Page](https://www.cgl.ucsf.edu/chimera/)

<br>

**fetch**

<img width="400" height="747" alt="image" src="https://github.com/user-attachments/assets/9bb6808b-a44e-4c49-b263-8d52e4524559" />

USCF Chimera는 UniProt 코드를 이용해서 불러올 수 있다. <br>

<br>

****








