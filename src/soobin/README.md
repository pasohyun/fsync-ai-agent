# 유튜브 댓글 멀티라벨 분류 모델

유튜브 직캠 댓글을 7개 라벨로 분류하는 멀티라벨 텍스트 분류 모델입니다.  
베이스 모델: `klue/roberta-base` (한국어 특화 RoBERTa)

## 라벨

| 라벨 | 설명 |
|------|------|
| 섹시 | 외모/신체 반응 |
| 응원 | 팬 응원/지지 표현 |
| 입덕 | 신규 팬 선언 |
| 퍼포먼스 | 무대/댄스/노래/의상 평가 |
| 유머 | 밈/농담/유머 반응 |
| 재방문 | 반복 시청 언급 |
| 기타 | 위 항목에 해당하지 않는 댓글 |

## 프로젝트 구조

```
subin2/
├── data/
│   ├── comments.jsonl           # 원본 크롤링 데이터
│   ├── comments_korean.jsonl    # 한국어 필터링 결과
│   ├── comments_sexy.jsonl      # 섹시 라벨 필터
│   ├── to_label.jsonl           # 라벨링 대기 데이터
│   ├── comments_labeled.jsonl   # 라벨링 완료 데이터 (전체)
│   ├── train.jsonl              # 학습용 (70%)
│   └── val.jsonl                # 평가용 (30%)
│
├── data_script/
│   ├── youtube.py               # 유튜브 댓글 크롤링
│   ├── filter_korean.py         # 한국어 필터링
│   ├── label_comments.py        # Claude API 자동 라벨링
│   └── apply_labels.py          # 수동 라벨 적용
│
├── model_training_script/
│   ├── config.py                # 하이퍼파라미터, 라벨 목록, 경로
│   ├── dataset.py               # PyTorch Dataset 클래스
│   ├── model.py                 # RoBERTa + 분류 헤드
│   ├── split_data.py            # 데이터 7:3 분할
│   ├── train.py                 # 학습 루프
│   ├── eval.py                  # 평가 (micro/macro F1)
│   └── predict.py               # 추론 (댓글 → 라벨 확률)
│
├── model_output/
│   └── best_model.pt            # 저장된 best 모델
│
└── subinn/                      # 가상환경
```

## 실행 순서

```bash
# 가상환경 활성화
source subinn/bin/activate

cd model_training_script

# 1. 데이터 분할 (train 70% / val 30%)
python split_data.py

# 2. 학습
python train.py

# 3. 평가
python eval.py

# 4. 추론 (대화형)
python predict.py
```

## 데이터 형식

```json
{
  "text": "진짜 너무 예쁘다 완전 내 스타일이야",
  "labels": ["섹시", "입덕"]
}
```

## 모델 구조

- **Encoder**: `klue/roberta-base` ([CLS] 토큰 벡터 사용)
- **Head**: Dropout(0.1) → Linear(768 → 7)
- **Loss**: `BCEWithLogitsLoss` (라벨별 독립 이진 분류)
- **추론**: `sigmoid` → threshold 0.5 이상이면 해당 라벨 양성

## 주요 하이퍼파라미터

| 항목 | 값 |
|------|----|
| 모델 | klue/roberta-base |
| max_len | 128 |
| batch_size | 32 |
| epochs | 5 |
| learning_rate | 2e-5 |
| threshold | 0.5 |

## 필요 패키지

```bash
pip install transformers torch scikit-learn tqdm
```
