# config/labels.py
# 팀 공통 라벨 정의 — 이 파일 기준으로 통일

# ── Track A: RoBERTa 학습용 (범용) ──────────────
TRAIN_LABELS = [
    "cute",       # 귀여움 — 발랄하고 사랑스러운 무드
    "fresh",      # 청량 — 밝고 시원한 에너지
    "hip",        # 힙 — 강하고 스트릿한 에너지
    "sexy",       # 섹시 — 성숙하고 매혹적인 무드
    "emotional",  # 감성/서정 — 잔잔하고 감동적인 무드
    "powerful",   # 파워풀 — 압도적이고 에너지 넘치는 무드
    "dark",       # 다크/미스터리 — 어둡고 몽환적인 분위기
    "other",      # 기타/판단불가 — 컨셉 무관 댓글 (응원, 팬심 등)
]

# ── Track B: Gap 분석용 (엔플라잉 전용) ──────────
NFLYING_LABELS = [
    "band_identity",   # 밴드/락 정체성 — "역시 밴드", "기타 소리 미쳐"
    "emotional_ballad",# 감성/발라드 — "눈물난다", "드라마 OST 느낌"
    "fresh_bright",    # 청량/밝음 — "기분 좋아진다", "여름 노래"
    "growth_narrative",# 성장/서사 — "이 팀 오래 봐왔는데", "데뷔 때부터"
    "new_fan",         # 신규 유입 — "드라마 보고 왔다", "처음 알았는데"
    "other",           # 기타
]

# ── purpose 값 ───────────────────────────────────
PURPOSE = {
    "TRAIN":      "train",       # RoBERTa 학습용
    "GAP_TARGET": "gap_target",  # Gap 분석 대상 (엔플라잉)
}

# ── video_type 값 ────────────────────────────────
VIDEO_TYPE = {
    "MV":      "MV",
    "TEASER":  "teaser",
    "FANCAM":  "fancam",
    "BEHIND":  "behind",
    "LIVE":    "live",
}
