# 유튜브 댓글 멀티라벨 분류 및 Gap 분석

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

발명내용 설명서 [도면 1] 전체 파이프라인 구성도(S110~S170)의 각 "부"와
디렉터리가 1대1로 대응한다. 저장소에는 S110~S170 코드만 포함하며,
단계별 입출력 데이터(`01_corpus/` `02_intent/` `04_reaction/` `05_최종결과/`)는
용량 문제로 저장소에 포함하지 않고 로컬에만 보관한다.

```
subin2/
│
├── S110_반응데이터_수집정제부/          # 반응 데이터 수집·정제부
│   ├── youtube.py                     # YouTube Data API 수집 (키는 환경변수)
│   ├── filter_korean.py               # 언어 비율 필터
│   ├── merge_corpus.py                # 통합·중복 제거
│   └── crawlers/band/Band_crawler_v1.ipynb  # 해시 중복 제거·스팸 패턴 필터 포함
│
├── S120_반응라벨체계_유도부/            # 반응 라벨 체계(Taxonomy) 유도부
│   └── label_descriptions.json        # 라벨별 정의·키워드 (12라벨)
│                                      # ※ 임베딩·클러스터링 유도 코드는 미구현
│
├── S130_이중라벨링_분류기학습부/        # 이중 라벨링·분류기 학습부
│   ├── labeling/
│   │   ├── label_corpus_batch.py      # LLM 약지도 라벨링 (Batch API)
│   │   ├── claude_label_gold_600.py   # 골드셋 blind 라벨링 (품질 게이트)
│   │   ├── merge_gold.py              # 골드셋 병합
│   │   └── make_labeling_sheet.py     # 수동 라벨링 시트 생성
│   ├── model_training_script/         # ME5 + 시그모이드 이진 분류 헤드
│   │   ├── config.py  dataset.py  model.py  split_data.py
│   │   ├── train.py   eval.py     predict.py
│   │   └── train.jsonl  val.jsonl
│   └── model_output/                  # 모델 체크포인트 · 평가 로그
│       ├── me5_large_v2/              # 최종 채택 모델 ★ (best_model.pt)
│       └── me5_large/ deberta_large/ modernbert_large/ roberta_large/ human/
│                                      # 모델 비교 실험 — eval 로그만 보존
│
├── S140_기준선반응분포_산출부/          # 기준선 반응 분포(Baseline) 산출부
│   ├── run_corpus_sigmoid.py          # 전체 코퍼스 107,505건 · τ=0.80
│   ├── run_all_artists_sigmoid.py     # 아티스트별 (동종 코호트 비교용)
│   └── run_nflying_sigmoid.py         # 엔플라잉 단독
│
├── S150_대상반응분포_산출부/            # 대상 반응 분포(Target) 산출부
│   └── run_hwanjeolgi_sigmoid.py      # 환절기 릴리즈 4,000건 · 동일 τ
│
├── S160_의도라벨_추출부/                # 의도 라벨 추출부
│   └── step1_build_intent_vectors.py  # 의도 level(high/med/low) → 점수 변환
│
├── S170_괴리도산출_리포팅부/            # 괴리도(Gap) 산출·리포팅부
│   ├── compute_gap_lift.py            # lift = 대상 분포 ÷ 기준선 분포, 판정
│   ├── make_figures.py                # 도면 1·2 및 추이 그래프 생성
│   └── 대시보드/                        # Streamlit 웹 대시보드
│       ├── app.py
│       └── data/최종 데이터/
│
├── subinn/                            # Python 가상환경 (git 제외)
├── requirements.txt
└── README.md
```

### 로컬 데이터 디렉터리 (git 제외)

스크립트는 저장소 루트 기준 아래 경로를 참조한다. 클론 후 실행하려면
이 디렉터리들을 별도로 확보해야 한다.

```
01_corpus/                             # [S110·S130] 코퍼스·골드셋
├── corpus_labeled_1.jsonl             #   8개 아티스트 107,935행 (한국어 107,505)
├── gold_680_final.jsonl               #   인간 검증 골드셋 680건
├── artist_raw/                        #   아티스트별 원본 크롤링
└── labeling_sheets/                   #   팀원 라벨링 시트

02_intent/                             # [S160] 기획 문서·의도 라벨
├── texts/  extracted/                 #   기획서 원문 TXT · 라인별 추출 JSON
├── intent_labels/                     #   ★ 의도 라벨 + 강조 강도
│   └── hwanjeolgi_intent.json         #     compute_gap_lift.py 의 --intent 입력
└── _archive/                          #   구 릴리즈 의도 프로필 · 원본 PDF(사내 문서)

04_reaction/                           # [S140·S150·S170] 반응 분포·괴리도
├── 전체/OUTPUT/                        #   기준선 분포 (corpus_reaction_*)
├── 아티스트별/                          #   코호트 분포 (8팀)
└── 환절기/
    ├── output/                        #   대상 분포 (hwanjeolgi_reaction_*)
    └── lift/                          #   ★ 괴리도 (hwanjeolgi_gap_lift_*)

05_최종결과/                            # 발표용 산출물
└── 도면/                               #   ★ 설명서 도면 1·2 + 추이 그래프
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
python S140_기준선반응분포_산출부/run_corpus_sigmoid.py

# [S150] 대상 반응 분포 — 환절기 릴리즈 댓글 4,000건
python S150_대상반응분포_산출부/run_hwanjeolgi_sigmoid.py

# 동종 코호트 포지셔닝용 — 아티스트별 반응 분포
python S140_기준선반응분포_산출부/run_all_artists_sigmoid.py

# [S170] 괴리도(lift) 산출 — 의도 라벨은 02_intent/intent_labels/ 에서 읽음
python S170_괴리도산출_리포팅부/compute_gap_lift.py

# 기존 산출물과 일치하는지 검증 (재현성 확인)
python S170_괴리도산출_리포팅부/compute_gap_lift.py \
    --verify 04_reaction/환절기/lift/hwanjeolgi_gap_lift_080_9label.json

# 도면 생성
python S170_괴리도산출_리포팅부/make_figures.py --fig 1 --fig 2 --fig trend

# 대시보드
streamlit run S170_괴리도산출_리포팅부/대시보드/app.py
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
> `run_corpus_sigmoid.py`(S140) 와 `run_hwanjeolgi_sigmoid.py`(S150) 를 **함께** 재실행할 것.

---

## 괴리도 판정 체계

`compute_gap_lift.py` 의 판정 기준. 제1 기준값 1.0을 의도·비의도 라벨에 공통 적용한다.

| 구분 | 상대 강도 | 판정 |
|------|-----------|------|
| 의도 라벨 | ≥ 1.0 (제1 기준값) | ✅ 정렬 — 의도한 메시지가 전달됨 |
| 의도 라벨 | < 1.0 | 🔴 누수 — 전달되지 못함 |
| 비의도 라벨 | ≥ 1.5 (제2 기준값) | 🟡 자생 — 의도하지 않은 반응이 강하게 발생 |
| 비의도 라벨 | 1.0 이상 ~ 1.5 미만 | 　보통 — 통상 수준 초과 |
| 비의도 라벨 | < 1.0 | 　낮음 — 통상 수준 미달 |

기준값은 `--align-cut`(제1) / `--high-cut`(제2) / `--low-cut` 으로 조정 가능하다.
`gap_scalar = 1 − (전달된 의도 라벨 수 ÷ 전체 의도 라벨 수)`.
