# 유튜브 댓글 멀티라벨 분류 및 Gap 분석

## 팀 정보

| 항목 | 내용 |
|------|------|
| 팀명 | 심사숙고 (8팀) |
| 기간 | 2026-05-26 ~ 2026-07-03 (6주) |
| 팀장 | 유주형 |
| 팀원 | 강수빈, 박소현, 정지민 |

---

## 프로젝트 개요

K-밴드 아티스트 YouTube 댓글을 12개 반응 라벨로 분류하는 멀티라벨 모델을 학습하고,  
기획서의 의도 벡터와 실제 팬 반응 벡터를 비교하는 **Gap 분석** 파이프라인을 구축합니다.

- **베이스 모델**: `intfloat/multilingual-e5-large` (ME5 large v2 fine-tuned)
- **분류 방식**: Sigmoid 독립 이진 분류 (BCEWithLogitsLoss)
- **분석 대상**: 엔플라잉(N.Flying) 환절기 기획서 vs 팬 댓글 반응

---

## 반응 라벨 (12개)

| 라벨 | 설명 |
|------|------|
| 기타_노이즈 | 분류 불가 댓글 |
| 비주얼_멤버매력 | 아티스트 외모/이미지 반응 |
| 장기_팬덤 | 오랜 팬 결속, 10주년, 영원 |
| 역주행_기대 | 역주행 기대/언급 |
| 보컬_라이브 | 보컬·라이브 퍼포먼스 반응 |
| 신규_유입 | 신규 팬 입덕 선언 |
| 밴드_정체성 | 락·인디 밴드 아이덴티티 |
| 위로_공감 | 감정 공감, 위로, 낭만 |
| 연주_악기 | 기타·드럼 등 악기 언급 |
| 이별_감성 | 이별·상실·건조함 감성 |
| 청량_여름 | 청량감·여름·설렘 반응 |
| 음악성 | 미디엄 템포·리듬·라이브 중심 |

---

## 프로젝트 구조

```
subin2/
│
├── 01_corpus/                         # 원시 댓글 데이터
│   ├── corpus_labeled.jsonl           # 전체 라벨링 코퍼스 (v1)
│   ├── corpus_labeled_1.jsonl         # 전체 라벨링 코퍼스 (v2, 8개 아티스트 107,505개)
│   ├── gold_680_final.jsonl           # 골드 라벨 검증 세트 (680개)
│   ├── subin_labeling_sheet.xlsx      # 수빈 라벨링 시트
│   ├── artist_raw/                    # 아티스트별 원본 크롤링 JSONL
│   │   ├── data_cnblue/
│   │   ├── data_ftisland/
│   │   ├── data_qwer/
│   │   └── data_sexy_old/
│   └── labeling_sheets/               # 팀원 라벨링 시트
│       ├── jimin_labeling_sheet.xlsx
│       ├── juhyeong_labeling_sheet.xlsx
│       ├── sohyun_labeling_sheet.xlsx
│       └── subin_labeling_sheet.xlsx
│
├── 02_intent/                         # 기획서 의도 추출
│   ├── texts/                         # 기획서 원문 TXT (8개 문서)
│   ├── extracted/                     # 라인별 의도 추출 JSON (02_intent_extraction)
│   └── _archive/                      # 이전 버전 의도 벡터 (YAML/JSON)
│       ├── intent_vectors/            # 릴리즈별 intent YAML
│       ├── intent_12label/            # 12라벨 intent_vectors_12.json
│       └── raw/                       # 원본 PDF
│
├── 03_doc_probs/                      # 기획서 문서 라벨 확률
│   ├── finetuned/                     # Fine-tuned ME5 결과 (doc_probs_line, fulltext, wo_noise)
│   ├── puremodel/                     # Pure ME5 코사인 유사도 결과
│   └── _archive/                      # 초기 gap report JSON
│
├── 04_reaction/                       # 댓글 반응 분석 결과
│   ├── 환절기/                         # 엔플라잉 '환절기' 댓글 (4,000개)
│   │   ├── output/                    # sigmoid/softmax 추론 결과 + 분포 그래프
│   │   └── lift/                      # gap lift JSON (8label, 9label)
│   ├── 전체/                           # 전체 코퍼스 반응 (107,505개)
│   │   └── OUTPUT/                    # corpus_me5_sigmoid.jsonl + 반응파일 (7/8/9라벨)
│   └── 아티스트별/                     # 8개 아티스트 개별 반응 파일
│       ├── 엔플라잉/
│       ├── CNBLUE/
│       ├── Day6/
│       ├── FTISLAND/
│       ├── LUCY/
│       ├── QWER/
│       ├── 드래곤포니/
│       └── 엑스디너리히어로즈/
│
├── 05_최종결과/                        # 발표용 최종 산출물
│   ├── 반응추이(일별 그래프 9개)(히트수 라벨로 그래프 그리면됨)/
│   ├── 엔플라잉 포지셔닝(리액션퍼센트 라벨별로 비교하면됨)/
│   ├── 의도한 메세지별 반응 강도(위로,청량,장기팬덤만 사용하면됨)/
│   └── 워드클라우드/
│
├── data_script/                       # 데이터 수집·처리 스크립트
│   ├── youtube.py                     # YouTube 댓글 크롤링
│   ├── filter_korean.py               # 한국어 필터링
│   ├── label_comments.py              # Claude API 자동 라벨링
│   ├── apply_labels.py                # 수동 라벨 적용
│   ├── crawlers/
│   │   ├── band/Band_crawler_v1.ipynb # 밴드 크롤러
│   │   └── hip/hip_crawler_v2.ipynb   # 힙합 크롤러
│   └── labeling/                      # 라벨링 보조 스크립트
│
├── gap_analysis/                      # Gap 분석 파이프라인 스크립트
│   ├── step1_build_intent_vectors.py  # 기획서 의도 → float 벡터
│   ├── step2_build_reaction_vector.py # 댓글 반응 벡터 생성
│   ├── step3_compute_gap.py           # Gap score 산출
│   ├── probe_pure_me5.py              # Pure ME5 코사인 유사도 탐색
│   ├── probe_intent_json.py           # 의도 JSON 확률 탐색
│   ├── run_corpus_sigmoid.py          # 전체 코퍼스 sigmoid 추론
│   ├── run_hwanjeolgi_both.py         # 환절기 sigmoid+softmax 추론
│   ├── run_nflying_sigmoid.py         # 엔플라잉 단독 sigmoid 추론
│   ├── run_all_artists_sigmoid.py     # 전체 아티스트 sigmoid 추론
│   └── label_descriptions.json        # 라벨별 키워드 설명
│
├── model_training_script/             # 모델 학습 파이프라인
│   ├── config.py                      # 하이퍼파라미터, 라벨 목록, 경로
│   ├── dataset.py                     # PyTorch Dataset
│   ├── model.py                       # ME5 + 분류 헤드
│   ├── split_data.py                  # 데이터 분할
│   ├── train.py                       # 학습 루프
│   ├── eval.py                        # 평가 (micro/macro F1)
│   └── predict.py                     # 추론
│
├── model_output/                      # 모델 체크포인트
│   ├── me5_large_v2/                  # 최종 사용 모델 ★
│   ├── me5_large/
│   ├── deberta_large/
│   ├── modernbert_large/
│   ├── roberta_large/
│   └── human/                         # 휴먼 라벨 기준선
│
├── subinn/                            # Python 가상환경 (git 제외)
└── README.md
```

---

## 데이터 스키마

```json
{
  "video_id":           "string",
  "video_type":         "MV | teaser | fancam | behind | live",
  "video_title":        "string",
  "video_published_at": "ISO8601",
  "artist":             "string",
  "label":              "cute | fresh | hip | sexy | emotional | powerful | dark | other",
  "purpose":            "train | gap_target",
  "comment_id":         "string",
  "text":               "string",
  "likes":              "int",
  "published_at":       "ISO8601",
  "crawled_at":         "ISO8601"
}
```

---

## 실행 순서

```bash
# 가상환경 활성화
source subinn/bin/activate

# 1. 전체 코퍼스 sigmoid 추론 (GPU 0)
python gap_analysis/run_corpus_sigmoid.py

# 2. 환절기 댓글 sigmoid + softmax 추론 (GPU 1)
python gap_analysis/run_hwanjeolgi_both.py

# 3. 아티스트별 반응 파일 생성
python gap_analysis/run_all_artists_sigmoid.py

# 4. Gap 분석
python gap_analysis/step1_build_intent_vectors.py
python gap_analysis/step3_compute_gap.py --release hwanjeolgi
```

---

## 주요 하이퍼파라미터

| 항목 | 값 |
|------|----|
| 베이스 모델 | intfloat/multilingual-e5-large |
| 파인튜닝 버전 | me5_large_v2 |
| max_len | 128 |
| batch_size | 64 |
| 추론 방식 | sigmoid (독립 이진) |
| 반응 임계값 | 0.80 / 0.85 / top1 |
| 분석 라벨 수 | 9라벨 (기타_노이즈·비주얼·역주행 제외) |
