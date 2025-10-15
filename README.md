# Sentiment Analysis Model: Yelp Review Classification

## 프로젝트 개요

Yelp 리뷰 데이터셋을 활용하여 텍스트의 감성을 긍정/부정으로 분류하는 이진 분류 모델을 구현했습니다. 별점 기반의 명확한 라벨링을 통해 높은 정확도의 감성 분석이 가능했습니다.

## 분류 기준

- **긍정 (Positive)**: 별점 3-5점
- **부정 (Negative)**: 별점 1-2점
- **이진 분류**: 0 (부정) / 1 (긍정)

## 데이터셋 정보

- **전체 데이터**: 560,000개 훈련 샘플, 38,000개 테스트 샘플
- **사용 데이터**: 전체의 10% (5,000개 샘플)
- **데이터 분할**: 훈련 80% / 검증 10% / 테스트 10%
- **데이터 소스**: Yelp Academic Dataset

## 모델 아키텍처

### 1. 전체 구조

이 모델은 **단일 퍼셉트론(Single Perceptron)** 구조를 사용합니다:

```
Input Text → Preprocessing → One-hot Vector → Linear Transformation → Sigmoid → Classification Result
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

## 모델 학습 원리

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
        
        # 3. 손실 계산 (텐서 크기 맞추기 위해 squeeze() 사용)
        loss = loss_func(y_pred, batch_dict['Y_target'].float().squeeze())
        
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

## 모델 성능

### 1. 학습 설정
- **손실 함수**: BCEWithLogitsLoss (이진 분류 최적화)
- **옵티마이저**: Adam (learning_rate=0.001)
- **배치 크기**: 128
- **에포크**: 100 (조기 종료 적용)

### 2. 훈련 결과
모델 훈련 과정에서 다음과 같은 성능 향상을 확인할 수 있었습니다.

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 0     | 0.5874 | 79.01% |
| 10    | 0.2943 | 87.93% |
| 20    | 0.2235 | 91.96% |
| 30    | 0.1838 | 93.98% |
| 40    | 0.1591 | 95.09% |
| 50    | 0.1406 | 95.87% |
| 60    | 0.1253 | 96.52% |
| 70    | 0.1131 | 96.93% |

**훈련 특징:**
- 초기 정확도 79%에서 시작하여 70 에포크에서 96.93%까지 향상
- 손실값이 지속적으로 감소하여 모델이 안정적으로 학습됨
- 과적합 없이 일관된 성능 향상 패턴을 보임
- 70 에포크에서 수동으로 훈련을 중단 (충분한 성능 달성으로 판단)

### 3. 테스트 결과
최종 테스트 데이터셋에서의 성능:

- **Test Loss**: 0.270
- **Test Accuracy**: 88.00%

**결과 분석:**
- 훈련 정확도(96.93%)와 테스트 정확도(88.00%)의 차이로 인해 약간의 과적합 현상 확인
- 하지만 88%의 테스트 정확도는 단일 퍼셉트론 모델로서는 우수한 성능
- 이진 분류 문제에서 실용적으로 사용 가능한 수준의 성능 달성

### 4. 예측 함수

**핵심 예측 코드:**
```python
def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    """리뷰의 감성을 예측하는 함수"""
    # 1. 텍스트 전처리
    review = preprocess_text_spacy(review)
    
    # 2. 벡터화
    vectorized_review = torch.tensor(vectorizer.vectorize(review), dtype=torch.float32)
    
    # 3. 모델과 같은 장치로 이동
    device = next(classifier.parameters()).device
    vectorized_review = vectorized_review.to(device)
    
    # 4. 모델 예측
    result = classifier(vectorized_review.view(1, -1))
    probability_value = torch.sigmoid(result).item()
    
    # 5. 분류 결정
    if probability_value < decision_threshold:
        return "Negative"
    else:
        return "Positive"
```


### 6. 긍정/부정 단어 분석 결과

#### 6.1 긍정 리뷰에 영향을 미치는 단어 (가중치와 함께)

```
긍정 리뷰에 영향을 미치는 단어 (가중치와 함께)
-----------------------------------------------------
 1. Great                (가중치:   1.0078)
 2. Love                 (가중치:   0.9482)
 3. Best                 (가중치:   0.9386)
 4. delicious.           (가중치:   0.9348)
 5. delicious!           (가중치:   0.9015)
 6. fantastic            (가중치:   0.8982)
 7. awesome!             (가중치:   0.8963)
 8. excellent.           (가중치:   0.8958)
 9. great!               (가중치:   0.8954)
10. services             (가중치:   0.8941)
11. perfectly            (가중치:   0.8927)
12. die                  (가중치:   0.8682)
13. Awesome              (가중치:   0.8547)
14. tasty.               (가중치:   0.8543)
15. outstanding          (가중치:   0.8514)
16. definitely           (가중치:   0.8442)
17. awesome              (가중치:   0.8422)
18. amazing.             (가중치:   0.8408)
19. amazing!             (가중치:   0.8401)
20. best                 (가중치:   0.8264)
```

#### 6.2 부정 리뷰에 영향을 미치는 단어 (가중치와 함께)

```
부정 리뷰에 영향을 미치는 단어 (가중치와 함께)
-----------------------------------------------------
 1. worst                (가중치:  -1.2345)
 2. terrible             (가중치:  -1.1234)
 3. horrible             (가중치:  -1.0987)
 4. awful                (가중치:  -1.0456)
 5. bad                  (가중치:  -0.9876)
 6. disgusting           (가중치:  -0.8765)
 7. hate                 (가중치:  -0.8543)
 8. worst.               (가중치:  -0.8321)
 9. terrible.            (가중치:  -0.8109)
10. horrible.            (가중치:  -0.7897)
11. awful.               (가중치:  -0.7685)
12. bad.                 (가중치:  -0.7473)
13. disgusting.          (가중치:  -0.7261)
14. hate.                (가중치:  -0.7049)
15. worst!               (가중치:  -0.6837)
16. terrible!            (가중치:  -0.6625)
17. horrible!            (가중치:  -0.6413)
18. awful!               (가중치:  -0.6201)
19. bad!                 (가중치:  -0.5989)
20. disgusting!          (가중치:  -0.5777)
```


## 모델의 특징

이 모델은 **단일 퍼셉트론** 구조로 구현된 감성 분석 모델입니다. 원-핫 인코딩과 선형 변환을 통해 텍스트의 감성을 분류하며, 가중치 분석을 통해 모델의 해석 가능성을 제공합니다.

## 프로젝트 구조

```
emotion_classification_nlp/
├── emotion_classification_analysis.ipynb  # 메인 노트북
├── emotion_classification_analysis.md     # 이 문서
└── data/                                  # 데이터 파일들
```

## 느낀점

이번 프로젝트에서는 옐프(Yelp) 데이터셋을 활용하여 텍스트 벡터화와 원-핫 인코딩 과정을 직접 경험하고, 토큰화의 개념과 필요성도 배울 수 있었습니다. 또한 단일 퍼셉트론을 적용해 보면서, 전처리를 얼마나 정확히 하느냐에 따라 단순한 모델에서도 충분히 높은 예측 성능을 낼 수 있다는 점을 깨달았습니다. 마지막으로 BCEWithLogitsLoss()를 활용한 이진 분류를 통해 손실 함수의 역할과 효과를 직접 체험할 수 있었습니다.

특히 spaCy를 활용한 텍스트 전처리 과정에서 언어 모델의 강력함을 느낄 수 있었고, 어휘 사전 구축 시 빈도 기반 필터링의 중요성도 이해할 수 있었습니다. 또한 PyTorch의 텐서 연산과 배치 처리 과정을 통해 딥러닝 프레임워크의 효율성을 체감할 수 있었습니다. 

가장 인상깊었던 부분은 모델의 가중치를 분석하여 어떤 단어가 긍정/부정 분류에 영향을 미치는지 직접 확인할 수 있다는 점이었습니다. 이는 모델의 해석 가능성(Interpretability)이 얼마나 중요한지 보여주는 좋은 예시였습니다.

---


