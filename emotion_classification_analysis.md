# 감성 분석 모델: Yelp 리뷰 긍정/부정 분류

## 📋 프로젝트 개요

Yelp 리뷰 데이터셋을 활용하여 텍스트의 감성을 긍정/부정으로 분류하는 이진 분류 모델을 구현했습니다. 별점 기반의 명확한 라벨링을 통해 높은 정확도의 감성 분석이 가능했습니다.

## 🎯 분류 기준

- **긍정 (Positive)**: 별점 3-5점
- **부정 (Negative)**: 별점 1-2점
- **이진 분류**: 0 (부정) / 1 (긍정)

## 📊 데이터셋 정보

- **전체 데이터**: 560,000개 훈련 샘플, 38,000개 테스트 샘플
- **사용 데이터**: 전체의 10% (5,000개 샘플)
- **데이터 분할**: 훈련 80% / 검증 10% / 테스트 10%
- **데이터 소스**: Yelp Academic Dataset

## 🧠 모델 아키텍처

### 1. 전체 구조

이 모델은 **단일 퍼셉트론(Single Perceptron)** 구조를 사용합니다:

```
입력 텍스트 → 전처리 → 원-핫 벡터 → 선형 변환 → 시그모이드 → 분류 결과
```

### 2. 핵심 구성 요소

#### 2.1 텍스트 전처리

**핵심 전처리 함수:**
```python
import spacy
import re

# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

def preprocess_text_spacy(text):
    """spaCy를 사용한 텍스트 전처리"""
    doc = nlp(text.lower())  # 소문자 변환 + 토큰화
    out = []
    for tok in doc:
        if tok.is_punct:  # 구두점 처리
            out.append(f" {tok.text} ")
        elif tok.is_alpha:  # 알파벳만 추출
            out.append(f" {tok.text} ")
    s = "".join(out)
    s = re.sub(r"\s+", " ", s).strip()  # 공백 정리
    return s
```

**처리 과정:**
1. **소문자 변환**: `text.lower()`
2. **토큰화**: spaCy의 언어 모델 사용
3. **구두점 처리**: 문장부호를 공백으로 분리
4. **알파벳 필터링**: 숫자, 특수문자 제거
5. **공백 정리**: 연속된 공백을 하나로 통합

**예시:**
- 입력: `"This restaurant is AMAZING!!! The food was delicious."`
- 출력: `"this restaurant is amazing the food was delicious"`

#### 2.2 어휘 사전 구축

**Vocabulary 클래스:**
```python
from collections import Counter
import string

class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        self._token_to_idx = dict(token_to_idx) if token_to_idx else {}
        self._idx_to_token = {i: t for t, i in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token
        self._unk_index = None
        
        if self._add_unk:
            self._unk_index = self.add_token(self._unk_token)
    
    def add_token(self, token):
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        idx = len(self._token_to_idx)
        self._token_to_idx[token] = idx
        self._idx_to_token[idx] = token
        return idx
    
    def lookup_token(self, token):
        return self._token_to_idx.get(token, self._unk_index)
    
    def lookup_index(self, index):
        return self._idx_to_token.get(index, self._unk_token)
```

**어휘 사전 구축 과정:**
```python
# 1. 단어 빈도 계산
word_counts = Counter()
for review in review_df.text:
    for word in review.split(" "):
        if word not in string.punctuation:
            word_counts[word] += 1

# 2. 빈도 기반 필터링 (cutoff=25)
review_vocab = Vocabulary()
for word, count in word_counts.items():
    if count > cutoff:  # 25번 이상 등장한 단어만
        review_vocab.add_token(word)
```

**핵심 특징:**
- **빈도 필터링**: 25회 이상 등장하는 단어만 포함
- **UNK 토큰**: 미지 단어 처리
- **양방향 매핑**: 단어↔인덱스 변환 지원

#### 2.3 원-핫 인코딩

**ReviewVectorizer 클래스:**
```python
import numpy as np
import string

class ReviewVectorizer(object):
    def __init__(self, review_vocab):
        self.review_vocab = review_vocab
    
    def vectorize(self, review):
        """텍스트를 원-핫 벡터로 변환"""
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float64)
        for token in review.split(" "):
            if token not in string.punctuation:
                word_index = self.review_vocab.lookup_token(token)
                one_hot[word_index] = 1  # 해당 위치에 1 할당
        return one_hot
    
    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """데이터프레임에서 벡터라이저 생성"""
        review_vocab = Vocabulary()
        
        # 단어 빈도 계산
        word_counts = Counter()
        for review in review_df.text:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1
        
        # 빈도 기반 필터링
        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)
        
        return cls(review_vocab)
```

**벡터화 과정:**
```python
# 어휘 사전: {"pizza": 0, "fantastic": 1, "terrible": 2, "good": 3, "bad": 4}
# 문장: "pizza fantastic good"
# 원-핫 벡터: [1, 1, 0, 1, 0]
```

**핵심 특징:**
- **희소 벡터**: 대부분 0, 일부만 1인 벡터
- **고차원**: 어휘 사전 크기만큼의 차원
- **단어 존재 여부**: 단어가 있으면 1, 없으면 0

#### 2.4 모델 구조
```python
class ReviewClassifier(nn.Module):
    def __init__(self, num_features):
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)
    
    def forward(self, x_in, apply_sigmoid=False):
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out
```

**모델 특징:**
- **입력층**: 어휘 사전 크기만큼의 원-핫 벡터
- **출력층**: 1개 뉴런 (이진 분류)
- **활성화 함수**: 시그모이드 (확률 출력)

## 🧠 모델 학습 원리

### 1. 학습 과정

**핵심 학습 코드:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 손실 함수 및 옵티마이저 설정
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# 학습 루프
for epoch_index in range(num_epochs):
    dataset.set_split('train')
    batch_generator = generate_batches(dataset, batch_size=128, device=device)
    
    for batch_index, batch_dict in enumerate(batch_generator):
        # 1. 기울기 초기화
        optimizer.zero_grad()
        
        # 2. 순전파
        y_pred = classifier(x_in=batch_dict['X_data'].float())
        
        # 3. 손실 계산
        loss = loss_func(y_pred, batch_dict['Y_target'].float())
        
        # 4. 역전파
        loss.backward()
        
        # 5. 가중치 업데이트
        optimizer.step()
```

이 모델은 **지도학습(Supervised Learning)**을 통해 학습됩니다:

1. **순전파**: 입력 텍스트 → 원-핫 벡터 → 선형 변환 → 시그모이드 → 예측값
2. **손실 계산**: BCEWithLogitsLoss를 사용하여 예측값과 실제 라벨 비교
3. **역전파**: 손실에 대한 기울기 계산
4. **가중치 업데이트**: Adam 옵티마이저로 가중치 조정

### 2. 가중치 학습 메커니즘

#### 초기 상태 (랜덤 가중치)
```
단어별 가중치: {"pizza": 0.1, "fantastic": -0.3, "terrible": 0.2, "good": -0.1, "bad": 0.4}
```

#### 학습 과정
1. **예측**: 긍정 리뷰 "pizza fantastic good" → 가중치 합계 = 0.1 + (-0.3) + (-0.1) = -0.3
2. **확률 변환**: 시그모이드(-0.3) = 0.43 < 0.5 → "부정" (틀림!)
3. **손실 계산**: BCE 손실로 큰 오차 감지
4. **가중치 조정**: 긍정 단어들의 가중치 증가, 부정 단어들의 가중치 감소

#### 최종 상태 (학습 완료)
```
단어별 가중치: {"pizza": 0.8, "fantastic": 0.9, "terrible": -0.6, "good": 0.7, "bad": -0.5}
```

### 3. 모델의 해석 가능성

이 모델의 가장 큰 장점은 **해석 가능성**입니다:

- **긍정 단어**: 양수 가중치 (fantastic: 0.9, good: 0.7)
- **부정 단어**: 음수 가중치 (terrible: -0.6, bad: -0.5)
- **중립 단어**: 0에 가까운 가중치

## 📈 모델 성능

### 1. 학습 설정
- **손실 함수**: BCEWithLogitsLoss (이진 분류 최적화)
- **옵티마이저**: Adam (learning_rate=0.001)
- **배치 크기**: 128
- **에포크**: 100 (조기 종료 적용)

### 2. 예측 함수

**핵심 예측 코드:**
```python
def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    """리뷰의 감성을 예측하는 함수"""
    # 1. 텍스트 전처리
    review = preprocess_text_spacy(review)
    
    # 2. 벡터화
    vectorized_review = torch.tensor(vectorizer.vectorize(review), dtype=torch.float32)
    
    # 3. 모델 예측
    result = classifier(vectorized_review.view(1, -1))
    probability_value = torch.sigmoid(result).item()
    
    # 4. 분류 결정
    if probability_value < decision_threshold:
        return "Negative"
    else:
        return "Positive"
```

### 3. 예측 예시

**긍정 리뷰:**
```python
test_review = "this is a pretty awesome book"
prediction = predict_rating(test_review, classifier, vectorizer)
# 처리: awesome(0.8) + pretty(0.3) + good(0.7) = 1.8
# 확률: 시그모이드(1.8) = 0.86 > 0.5
# 결과: "Positive"
```

**부정 리뷰:**
```python
test_review = "terrible service and awful food"
prediction = predict_rating(test_review, classifier, vectorizer)
# 처리: terrible(-0.6) + awful(-0.4) + bad(-0.5) = -1.5
# 확률: 시그모이드(-1.5) = 0.18 < 0.5
# 결과: "Negative"
```

## 🎯 모델의 특징

### 장점
1. **단순성**: 복잡한 구조 없이 효과적인 분류
2. **해석 가능성**: 가중치로 단어의 영향력 확인 가능
3. **빠른 학습**: 단일 레이어로 빠른 수렴
4. **안정성**: 과적합 위험 낮음

### 한계
1. **단어 순서 무시**: "not good"과 "good"을 구분하지 못함
2. **문맥 부족**: 문장의 전체적인 의미 파악 한계
3. **희소성 문제**: 원-핫 인코딩으로 인한 고차원 벡터

## 🚀 활용 분야

- **리뷰 분석**: 온라인 쇼핑몰, 레스토랑 리뷰 감성 분석
- **소셜 미디어**: 트위터, 페이스북 감성 모니터링
- **고객 서비스**: 고객 피드백 자동 분류 및 우선순위 설정
- **마케팅**: 제품/서비스에 대한 고객 반응 실시간 분석

## 📁 프로젝트 구조

```
emotion_classification_nlp/
├── emotion_classification_analysis.ipynb  # 메인 노트북
├── emotion_classification_analysis.md     # 이 문서
└── data/                                  # 데이터 파일들
```

---

*이 프로젝트는 PyTorch를 활용한 텍스트 분류의 기본 원리를 보여주는 교육용 예제입니다.*
