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
│   │   └── data_qwer/
│   └── labeling_sheets/               # 팀원 라벨링 시트
│       ├── jimin_labeling_sheet.xlsx
│       ├── juhyeong_labeling_sheet.xlsx
│       ├── sohyun_labeling_sheet.xlsx
│       └── subin_labeling_sheet.xlsx
│
├── 02_intent/                         # 기획서 의도 추출
│   ├── texts/                         # 기획서 원문 TXT (8개 문서)
│   ├── extracted/                     # 라인별 의도 추출 JSON
│   ├── intent_labels/                 # ★ 의도 라벨 + 강조 강도 (설명서 5.(6) 산출물)
│   │   └── hwanjeolgi_intent.json
│   └── _archive/
│       ├── intent_12label/            # 12라벨 intent_vectors_12.json (구 릴리즈 3건)
│       └── raw/                       # 원본 PDF (사내 문서 — 외부 공유 금지)
│
├── 04_reaction/                       # 댓글 반응 분석 결과
│   ├── 환절기/                         # 엔플라잉 '환절기' 댓글 (4,000개)
│   │   ├── output/                    # sigmoid 추론 결과 + 반응 분포 JSON
│   │   └── lift/                      # ★ 괴리도(lift) JSON (8label, 9label)
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
│   ├── 워드클라우드/
│   └── 도면/                           # ★ 발명내용 설명서 도면 1·2 + 추이 그래프
│
├── 00_crawling/                       # 데이터 수집·라벨링 스크립트
│   ├── youtube.py                     # YouTube 댓글 크롤링 (API 키는 환경변수)
│   ├── filter_korean.py               # 한국어 비율 필터
│   ├── crawlers/band/Band_crawler_v1.ipynb
│   └── labeling/
│       ├── merge_corpus.py            # 팀원별 코퍼스 통합·중복 제거
│       ├── label_corpus_batch.py      # LLM 약지도 라벨링 (Batch API)
│       ├── claude_label_gold_600.py   # 골드셋 blind 라벨링 (라벨 품질 게이트)
│       ├── merge_gold.py              # 골드셋 병합
│       └── make_labeling_sheet.py     # 수동 라벨링 시트 생성
│
├── gap_analysis/                      # Gap 분석 파이프라인 스크립트
│   ├── run_corpus_sigmoid.py          # S140 기준선 반응 분포 (전체 코퍼스)
│   ├── run_all_artists_sigmoid.py     # 아티스트별 반응 분포 (코호트 비교용)
│   ├── run_nflying_sigmoid.py         # 엔플라잉 단독 반응 분포
│   ├── run_hwanjeolgi_sigmoid.py      # ★ S150 대상 반응 분포 (환절기)
│   ├── compute_gap_lift.py            # ★ S170 괴리도(lift) 산출·판정
│   ├── make_figures.py                # ★ 도면 1·2 및 추이 그래프 생성
│   ├── step1_build_intent_vectors.py  # 의도 level(high/med/low) → 점수 변환
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
├── model_output/                      # 모델 체크포인트 · 평가 로그
│   ├── me5_large_v2/                  # 최종 사용 모델 ★ (best_model.pt 보유)
│   ├── me5_large/                     # 이하 모델 비교 실험 — eval 로그만 보존
│   ├── deberta_large/                 #   (.pt 가중치는 용량 문제로 삭제)
│   ├── modernbert_large/
│   ├── roberta_large/
│   └── human/
│
├── subinn/                            # Python 가상환경 (git 제외)
├── requirements.txt
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

발명내용 설명서 5.의 단계 번호(S110~S170)와 대응한다.
모든 명령은 저장소 루트에서 실행한다 (스크립트가 상대 경로를 사용).

```bash
# 가상환경 활성화
source subinn/bin/activate
pip install -r requirements.txt

# [S140] 기준선 반응 분포 — 8팀 전체 코퍼스 107,505건
python gap_analysis/run_corpus_sigmoid.py

# [S150] 대상 반응 분포 — 환절기 릴리즈 댓글 4,000건
python gap_analysis/run_hwanjeolgi_sigmoid.py

# 동종 코호트 포지셔닝용 — 아티스트별 반응 분포
python gap_analysis/run_all_artists_sigmoid.py

# [S170] 괴리도(lift) 산출 — 의도 라벨은 02_intent/intent_labels/ 에서 읽음
python gap_analysis/compute_gap_lift.py

# 기존 산출물과 일치하는지 검증 (재현성 확인)
python gap_analysis/compute_gap_lift.py \
    --verify 04_reaction/환절기/lift/hwanjeolgi_gap_lift_080_9label.json

# 도면 생성
python gap_analysis/make_figures.py --fig 1 --fig 2 --fig trend

# 대시보드
streamlit run 06_대시보드/app.py
```

---

## 주요 하이퍼파라미터

| 항목 | 값 |
|------|----|
| 베이스 모델 | intfloat/multilingual-e5-large |
| 파인튜닝 버전 | me5_large_v2 |
| max_len | 128 |
| batch_size | 64 |
| 추론 방식 | sigmoid (독립 이진, BCEWithLogitsLoss) |
| 반응 임계값 τ | **0.80** (0.85 / top1 산출물은 파라미터 비교용) |
| 분석 라벨 수 | **9라벨** (기타_노이즈·비주얼_멤버매력·역주행_기대 제외) |
| 정규화 | 라벨별 히트 수 ÷ 분석 대상 라벨 전체 히트 수 |
| 괴리도 | lift(l) = p_target(l) / p_baseline(l) — 기준선·대상 모두 동일 τ·동일 정규화 |

> 기준선 분포와 대상 분포는 **동일 라벨 체계·동일 분류기·동일 임계값·동일 정규화 방식**으로
> 산출되어야 상대 강도 대비가 성립한다. 임계값이나 라벨 집합을 바꾸면
> `run_corpus_sigmoid.py` 와 `run_hwanjeolgi_sigmoid.py` 를 **함께** 재실행할 것.
