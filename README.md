# F-Sync AI Agent — 프로젝트 폴더 구조

## 팀 정보
- 팀명: 심사숙고 (8팀)
- 기간: 2026-05-26 ~ 2026-07-03 (6주)
- 담당: 유주형(팀장), 강수빈, 박소현, 정지민

---

## 폴더 구조

```
fsync/
│
├── data/                          ← 모든 데이터
│   ├── raw/                       ← 수집 원본 (절대 수정 금지)
│   │   ├── train/                 ← RoBERTa 학습용 댓글
│   │   │   ├── cute/              ← 귀여움 (소현: 엔시티위시)
│   │   │   ├── fresh/             ← 청량 (지민: 엔플라잉, 투어스, 라이즈, 세븐틴)
│   │   │   ├── hip/               ← 힙 (주형: 코르티스, 엔시티127, 피원하모니 등)
│   │   │   ├── sexy/              ← 섹시 (수빈: 몬스타엑스)
│   │   │   ├── emotional/         ← 감성/서정 🆕
│   │   │   ├── powerful/          ← 파워풀/강렬 🆕
│   │   │   ├── dark/              ← 다크/미스터리 🆕
│   │   │   └── other/             ← 기타/판단불가 (노이즈 처리용) 🆕
│   │   │
│   │   └── gap_analysis/          ← Gap 분석용 (엔플라잉 전용)
│   │       ├── nflying_current/   ← 6/2 컴백 댓글 (MV/티저/직캠/비하인드)
│   │       ├── nflying_past/      ← 과거 컴백 비교군
│   │       ├── dcin/              ← 디시인사이드 엔플라잉 갤
│   │       ├── naver_trends/      ← 네이버 데이터랩 검색량
│   │       └── press_release/     ← 보도자료 텍스트 (공개본)
│   │
│   ├── processed/                 ← 전처리 완료 데이터
│   │   ├── train/                 ← 학습용 전처리본
│   │   └── gap_analysis/          ← Gap 분석용 전처리본
│   │
│   └── annotated/                 ← GPT 어노테이션 완료 데이터
│
├── src/                           ← 소스 코드
│   ├── crawlers/                  ← 크롤링 스크립트
│   │   ├── crawl_nctwish.py       ← 소현 담당
│   │   ├── crawl_nflying.py       ← 지민 담당 (Gap 분석 대상)
│   │   ├── crawl_fresh.py         ← 지민 담당 (학습용)
│   │   ├── crawl_hip.py           ← 주형 담당
│   │   ├── crawl_sexy.py          ← 수빈 담당
│   │   └── crawl_dcin.py          ← 디시인사이드
│   │
│   ├── nlp/                       ← NLP 모델 코드
│   │   ├── preprocess.py          ← 전처리 (형태소 분석, 불용어 처리)
│   │   ├── annotate.py            ← GPT 어노테이션
│   │   └── train_roberta.py       ← RoBERTa 학습
│   │
│   ├── gap/                       ← Gap 분석 엔진
│   │   ├── keyword_extract.py     ← 키워드 추출
│   │   ├── gap_score.py           ← Gap score 계산
│   │   └── gap_explain.py         ← 원인 분석 (키워드 분해, 시계열)
│   │
│   ├── dashboard/                 ← Streamlit UI
│   │   └── app.py
│   │
│   └── utils/                     ← 공통 유틸
│       ├── schema.py              ← 데이터 스키마 정의
│       └── api_quota.py           ← API 크레딧 모니터링
│
├── outputs/                       ← 최종 산출물
│   ├── reports/                   ← Gap 분석 리포트
│   ├── planning_docs/             ← LLM 생성 기획서
│   └── models/                    ← 학습된 모델 저장
│
├── docs/                          ← 문서
│   ├── meeting_notes/             ← 회의록
│   └── references/                ← 참고자료
│
└── config/                        ← 설정 파일
    ├── labels.py                  ← 라벨 정의 (팀 공통)
    └── schema.json                ← 데이터 스키마 (팀 공통)
```

---

## 파일 네이밍 규칙

### 데이터 파일 (.jsonl)
```
{아티스트}_{video_type}_{날짜}.jsonl

예시:
  nct_wish_MV_20260602.jsonl
  nflying_MV_20260602.jsonl
  nflying_teaser_20260530.jsonl
  nflying_past_20250101.jsonl
```

### 날짜 형식
- 파일명: `YYYYMMDD`
- 데이터 내부 필드: ISO 8601 → `2026-06-02T14:30:00Z`

---

## 데이터 스키마 (팀 공통 — 반드시 준수)

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

> ⚠️ `label`과 `purpose`는 **영문 소문자 고정**. 한글 절대 금지.

---

## 담당별 수집 목표

| 담당 | 라벨 | 대상 아티스트 | 목표 댓글 수 | 저장 경로 |
|------|------|------------|------------|---------|
| 소현 | cute | 엔시티위시 | 2~3만 | data/raw/train/cute/ |
| 지민 | fresh | 엔플라잉, 투어스, 라이즈, 세븐틴 | 2~3만 | data/raw/train/fresh/ |
| 지민 | gap_target | 엔플라잉 (6/2 컴백) | 최대한 | data/raw/gap_analysis/nflying_current/ |
| 수빈 | sexy | 몬스타엑스 | 2~3만 | data/raw/train/sexy/ |
| 주형 | hip | 코르티스, 엔시티127, 피원하모니 등 | 2~3만 | data/raw/train/hip/ |

---

## API 크레딧 관리

- YouTube Data API v3: 댓글 100개 = 1 unit, 하루 무료 10,000 unit
- 25,000개 수집 = 약 250 unit → **하루 무료 한도 안에서 가능**
- $300 크레딧은 무료 한도 초과분 → 아껴서 Gap 분석 우선 배정

---

## Git 규칙

```
# 커밋 메시지 형식
[담당자] 작업내용

예시:
  [소현] 엔시티위시 MV 댓글 8,000개 수집 완료
  [수빈] RoBERTa 전처리 스크립트 추가
  [주형] Streamlit 대시보드 기본 골격 구현
```

> ⚠️ `data/raw/` 폴더는 `.gitignore`에 추가 (용량 문제)
> ⚠️ API 키는 절대 커밋 금지 → 환경변수로만 관리
