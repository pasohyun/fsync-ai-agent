"""
F-Sync 환절기 Gap 분석 대시보드 — 최종 확정본
실행: streamlit run app.py

화면 구성 (위→아래):
  1. 상단 바 (곡명, 기간)
  2. 헤드라인 카드 2개 (가장 아쉬운 포인트 / 기대 이상의 호응) + 인용 댓글 내장
  3. 다음 컴백을 위한 제안
  4. 팬들이 많이 쓴 표현 — 워드클라우드, 강조 3종(정렬/누수/자생)만, 클릭 시 관련 댓글 펼침
  5. 라벨별 반응 추이 — 다중 선택 겹쳐보기 + 프로모션 일정 마커
  6. 같은 라벨 · 컴백 단계별 비교 (티저/뮤직비디오/음악방송/부가 콘텐츠)

디자인은 최소한으로만 잡아둠 — 색상 토큰(VERDICT_COLOR), 레이아웃 구조는 Claude Code에서
밴드/FNC 톤으로 자유롭게 다듬는 것을 전제로 함. 데이터 로직과 인터랙션 구조가 핵심.
"""
import json
import re
import glob
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 0. 환경 설정
# ──────────────────────────────────────────────
st.set_page_config(page_title="F-Sync · 환절기 Gap 분석", layout="wide")

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgun.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
if FONT_PATH:
    fm.fontManager.addfont(FONT_PATH)
    try:
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
    except Exception:
        pass

APP_DIR = Path(__file__).resolve().parent
DATA_ROOT = APP_DIR / "data" / "최종 데이터"
GAP_LIFT_PATH = DATA_ROOT / "의도한 메세지별 반응 강도(위로,청량,장기팬덤만 사용하면됨)" / "hwanjeolgi_gap_lift_080_9label.json"
DAILY_TREND_PATH = DATA_ROOT / "반응추이(일별 그래프 9개)(히트수 라벨로 그래프 그리면됨)" / "hwanjeolgi_reaction_080_9label_by_date.json"
POSITIONING_DIR = DATA_ROOT / "엔플라잉 포지셔닝(리액션퍼센트 라벨별로 비교하면됨)"
WORDCLOUD_PATH = DATA_ROOT / "워드클라우드" / "hwanjeolgi_me5_sigmoid.jsonl"

# ──────────────────────────────────────────────
# 0-1. 전역 디자인 시스템 — 다크브라운/레드 히어로, 베이지 배경, 알약 버튼
# ──────────────────────────────────────────────
BRAND_DARK = "#2B2A28"     # PPT 기본 다크 브라운
BRAND_RED = "#D8362C"      # PPT 레드 (액센트)
BRAND_BEIGE = "#FAF7F5"    # PPT 전역 배경
CARD_BORDER = "#CFC9C6"    # PPT 보더/디바이더
GOOD_GREEN = "#2E7D73"     # PPT 틸 (포지티브)
GOOD_GREEN_BG = "#DCECE8"  # PPT 틸 틴트
BAD_RED = "#D8362C"        # PPT 레드
BAD_RED_BG = "#FBEAE8"     # PPT 핑크 틴트
SUGGEST_BROWN = "#767370"  # PPT 중간 그레이브라운
SUGGEST_BROWN_BG = "#EAE1DD"  # PPT 중간 베이지

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {{
    font-family: 'Sora', 'Noto Sans KR', sans-serif !important;
}}
.stApp {{
    background-color: {BRAND_BEIGE};
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 10px rgba(43,42,40,0.06);
}}
.fsync-hero {{
    background: linear-gradient(120deg, {BRAND_DARK} 0%, {BRAND_RED} 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(43,42,40,0.25);
}}
.fsync-hero-kicker {{
    color: #CFC9C6;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.fsync-hero-title {{
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 10px;
}}
.fsync-hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.16);
    color: #EAE1DD;
    font-size: 12px;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.25);
}}
.stButton > button {{
    border-radius: 999px !important;
    font-weight: 500 !important;
    border: 1px solid {CARD_BORDER} !important;
    transition: all 0.15s ease;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(120deg, {BRAND_DARK} 0%, {BRAND_RED} 100%) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(216,54,44,0.35);
}}
.stButton > button[kind="secondary"] {{
    background: #FFFFFF !important;
    color: {SUGGEST_BROWN} !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {BRAND_RED} !important;
    color: {BRAND_RED} !important;
}}
.fsync-pill {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}}
hr {{ border-color: {CARD_BORDER} !important; }}
</style>
<div class="fsync-hero">
  <div class="fsync-hero-kicker">FNC Entertainment · F-Sync Gap 분석</div>
  <div class="fsync-hero-title">N.Flying · 환절기</div>
  <span class="fsync-hero-badge">2026.05.22 – 2026.06.25</span>
</div>
""", unsafe_allow_html=True)

ALL_9_LABELS = ["장기_팬덤", "보컬_라이브", "신규_유입", "밴드_정체성", "위로_공감",
                "연주_악기", "이별_감성", "청량_여름", "음악성"]
EMPHASIS_LABELS = ["청량_여름", "밴드_정체성", "위로_공감"]
WORDCLOUD_LABELS = EMPHASIS_LABELS + ["연주_악기"]

LABEL_DISPLAY = {
    "장기_팬덤": "장기 팬덤", "보컬_라이브": "보컬 라이브", "신규_유입": "신규 유입",
    "밴드_정체성": "밴드 정체성", "위로_공감": "위로 공감", "연주_악기": "연주 악기",
    "이별_감성": "이별 감성", "청량_여름": "청량 여름", "음악성": "음악성",
}

VERDICT_COLOR = {
    # 코랄=부정(누수), 틸=긍정(정렬/자생) — PPT 2색 시그널 시스템
    "정렬": {"line": "#2E7D73", "bg": "#DCECE8", "text": "#1A5550"},   # 틸 — 의도대로 통함
    "누수": {"line": "#D8362C", "bg": "#FBEAE8", "text": "#9A1F18"},   # 코랄 — 기대 미달
    "자생": {"line": "#6FB16A", "bg": "#EBF4EA", "text": "#2C6328"},   # 그린 — 기대 이상의 호응
    "보통": {"line": "#767370", "bg": "#EAE1DD", "text": "#4A4744"},
    "낮음": {"line": "#767370", "bg": "#EAE1DD", "text": "#4A4744"},
}
SERIES_COLOR = {
    "밴드_정체성": "#D8362C",   # FNC 레드
    "위로_공감":   "#E06C60",   # 코랄 변형
    "청량_여름":   "#2E7D73",   # 틸
    "연주_악기":   "#E6B64D",   # 골드
    "장기_팬덤":   "#185FA5",   # 블루
    "보컬_라이브": "#534AB7",   # 퍼플
    "신규_유입":   "#6FB16A",   # 그린
    "음악성":      "#767370",   # 그레이
    "이별_감성":   "#E06C60",   # 코랄 (위로공감과 유사 계열이나 차트에서 구분됨)
}

KILLINGVOICE_MARKER = "킬링보이스"

PROMO_EVENTS = [
    {"date": "2026-05-21", "label": "MV 티저①"},
    {"date": "2026-05-25", "label": "컨셉 포토(단체)"},
    {"date": "2026-05-26", "label": "컨셉 포토(개인)"},
    {"date": "2026-05-29", "label": "가사 스포일러"},
    {"date": "2026-06-01", "label": "MV 티저②"},
    {"date": "2026-06-02", "label": "MV 발매(컴백)"},
    {"date": "2026-06-04", "label": "MV 비하인드"},
    {"date": "2026-06-08", "label": "레코딩 비하인드"},
]


# ──────────────────────────────────────────────
# 1. 데이터 로드 (캐시)
# ──────────────────────────────────────────────
@st.cache_data
def load_gap_lift():
    with open(GAP_LIFT_PATH, encoding="utf-8") as f:
        gap = json.load(f)
    for label, d in gap["labels"].items():
        d["verdict_clean"] = d["verdict"].strip().replace("✅ ", "").replace("🔴 ", "").replace("🟡 ", "")
    return gap


@st.cache_data
def load_daily_trend():
    with open(DAILY_TREND_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_positioning():
    name_map = {
        "nflying": "N.Flying", "CNBLUE": "CNBLUE", "Day6": "DAY6", "FTISLAND": "FTISLAND",
        "LUCY": "LUCY", "QWER": "QWER", "드래곤포니": "드래곤포니", "엑스디너리히어로즈": "엑스디너리히어로즈",
    }
    artist_pct = {}
    for f in glob.glob(str(POSITIONING_DIR / "*.json")):
        fname = Path(f).stem.replace("_reaction_080_9label", "")
        artist = name_map.get(fname, fname)
        with open(f, encoding="utf-8") as fp:
            artist_pct[artist] = json.load(fp)["reaction_pct"]
    return artist_pct


@st.cache_data
def load_wordcloud_comments():
    with open(WORDCLOUD_PATH, encoding="utf-8") as f:
        comments = [json.loads(l) for l in f]
    return [c for c in comments if KILLINGVOICE_MARKER not in c["video_title"]]


# ──────────────────────────────────────────────
# 2. 컴백 단계 분류 (video_title 패턴 기반)
# ──────────────────────────────────────────────
def classify_stage(title: str) -> str:
    if "TEASER" in title or "SPOILER" in title or "It's coming" in title:
        return "티저"
    if title.strip().endswith("MV"):
        return "뮤직비디오"
    if any(k in title for k in ("방송", "KBS", "음악회", "Concert")):
        return "음악방송"
    return "부가 콘텐츠"


@st.cache_data
def compute_stage_reaction():
    comments = load_wordcloud_comments()
    stage_data = defaultdict(lambda: {"total": 0, "hits": Counter()})
    for c in comments:
        stage = classify_stage(c["video_title"])
        stage_data[stage]["total"] += 1
        for label in ALL_9_LABELS:
            if c["sigmoid"].get(label, 0) > 0.80:
                stage_data[stage]["hits"][label] += 1
    result = {}
    for stage, d in stage_data.items():
        result[stage] = {l: round(d["hits"][l] / d["total"] * 100, 2) if d["total"] else 0 for l in ALL_9_LABELS}
    return result


# ──────────────────────────────────────────────
# 3. 일별 데이터 정리
# ──────────────────────────────────────────────
@st.cache_data
def daily_series_by_label():
    daily = load_daily_trend()
    by_date = daily["by_date"]
    dates = sorted(by_date.keys())
    series = {label: [by_date[d]["hit_counts"].get(label, 0) for d in dates] for label in ALL_9_LABELS}
    return dates, series


# ──────────────────────────────────────────────
# 4. 워드클라우드 — 강조 3종 + 자생 1종만, 정규식 기반 자동 추출
# ──────────────────────────────────────────────
SENTIMENT_END = re.compile(
    r"(좋다|좋아요?|좋네|좋은데|최고[다예]?요?|미쳤다|미쳤어요?|소름|대박|짱이다|"
    r"감사합니다|사랑해요?|사랑합니다|행복하다|행복해요?|든든하다|뭉클하다|예쁘다|예뻐요?)"
)
NOISE_START = re.compile(r"^(아|어|음|와|오|진짜|정말|너무|그냥|근데|역시)\s*")
REPEAT_CONSONANT = re.compile(r"[ㅋㅎㅠㅜㅏㅓ]{2,}")
WEIRD_CHARS = re.compile(r"[^\w\sㄱ-ㅣ가-힣!?]")
TYPO_PATTERN = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]{2,}|[가-힣]{1,2}앙[가-힣]")
LOW_QUALITY = re.compile(r"(같아요?$|것\s*같|그럼까지|아진짜요|왜케)")


def clean_clause(clause: str) -> str:
    clause = REPEAT_CONSONANT.sub("", clause)
    clause = WEIRD_CHARS.sub("", clause)
    clause = NOISE_START.sub("", clause)
    return re.sub(r"\s+", " ", clause).strip()


def extract_sentiment_clauses(text: str, max_len: int = 18):
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
    results = []
    for clause in re.split(r"[\n.!?~]+", text):
        raw = clause.strip()
        if not raw or len(raw) > max_len + 8 or not SENTIMENT_END.search(raw):
            continue
        cleaned = clean_clause(raw)
        if not (3 <= len(cleaned) <= max_len):
            continue
        if LOW_QUALITY.search(cleaned) or TYPO_PATTERN.search(cleaned):
            continue
        if len(cleaned.split()) < 2:
            continue
        results.append(cleaned)
    return results


@st.cache_data
def extract_evidence_quotes():
    comments = load_wordcloud_comments()
    quotes = {}
    for label in WORDCLOUD_LABELS:
        hits = [c for c in comments if c["sigmoid"].get(label, 0) > 0.80]
        counter = Counter()
        for c in hits:
            for cl in extract_sentiment_clauses(c["text"]):
                counter[cl] += 1
        quotes[label] = {"n_hits": len(hits), "top": counter.most_common(6)}
    return quotes


# ──────────────────────────────────────────────
# 4-1. 워드클라우드용 키워드 빈도 — 문장이 아니라 단어 단위로 세야 빈도 편차가 의미를 가짐
# ──────────────────────────────────────────────
PARTICLE_SUFFIXES = sorted([
    "이에요", "예요", "입니다", "이다", "에서", "에게", "한테", "까지", "부터", "처럼", "보다",
    "이라서", "라서", "이라", "라고", "으로", "에는", "에도", "와", "과", "을", "를", "이", "가",
    "은", "는", "의", "도", "만", "로", "에",
], key=len, reverse=True)

KEYWORD_STOPWORDS = {
    # 부사/접속사
    "너무", "진짜", "정말", "그냥", "근데", "역시", "오늘", "우리", "제발", "완전", "약간", "조금",
    "엄청", "그리고", "그런데", "이거", "저거", "그거", "해서", "합니다", "했어요", "이렇게", "저렇게",
    "모든", "항상", "계속", "아직", "벌써", "이제", "바로", "다시", "많이", "엄청나게", "거의", "보다도",
    # 지시/대명사
    "사람", "우와", "아니", "당신", "저는", "나는", "내가", "제가", "우리는", "정도",
    "이게", "이거", "이런", "저런", "그런", "이리", "저리", "이렇", "저렇", "그렇",
    # 일반 감탄/반응 (내용 없음)
    "좋다", "좋아요", "좋고", "이제는", "느낌", "생각", "보고", "기분", "하고", "하는",
    "이걸", "저걸", "그걸", "있어", "없어", "되는", "한거", "하면",
    # 멤버 비공식 호칭 (실명으로 대체)
    "훈이",
}


def strip_particle(token: str) -> str:
    for suf in PARTICLE_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 2:
            return token[: -len(suf)]
    return token


@st.cache_data
def extract_keyword_frequencies():
    """9개 라벨 전체에서 키워드 빈도 추출 — 라벨별 특색 있는 단어를 폭넓게 확보"""
    comments = load_wordcloud_comments()
    result = {}
    for label in ALL_9_LABELS:
        hits = [c for c in comments if c["sigmoid"].get(label, 0) > 0.80]
        counter = Counter()
        for c in hits:
            text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", c["text"])
            tokens = re.findall(r"[가-힣]{2,}", text)
            seen_in_comment = set()
            for t in tokens:
                kw = strip_particle(t)
                if len(kw) < 2 or len(kw) > 8 or kw in KEYWORD_STOPWORDS:
                    continue
                seen_in_comment.add(kw)
            counter.update(seen_in_comment)
        result[label] = {"n_hits": len(hits), "top": counter.most_common(15)}
    return result


@st.cache_data
def keyword_related_snippets(label: str, keyword: str, limit: int = 4):
    comments = load_wordcloud_comments()
    hits = [c for c in comments if c["sigmoid"].get(label, 0) > 0.80 and keyword in c["text"]]
    out = []
    for c in hits[:limit]:
        t = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", c["text"])
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > 45:
            t = t[:45] + "…"
        out.append(t)
    return out


@st.cache_data
def keyword_snippets_broad(keyword: str, limit: int = 5):
    """4개 라벨 댓글 전체에서 keyword가 포함된 댓글만 반환.
    - keyword가 다른 단어의 일부로만 쓰인 경우 제외 (예: '청량'검색 시 '청량하고'만 나오는 댓글 제외)
    - 언급 빈도 많은 댓글(keyword 2회 이상) 우선 정렬
    """
    comments = load_wordcloud_comments()
    hits = [
        c for c in comments
        if any(c["sigmoid"].get(lb, 0) > 0.80 for lb in WORDCLOUD_LABELS)
        and keyword in c["text"]
    ]
    # keyword 등장 횟수 기준 내림차순 정렬
    hits.sort(key=lambda c: c["text"].count(keyword), reverse=True)
    out = []
    for c in hits[:limit * 4]:
        t = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", c["text"])
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) < 8:
            continue
        if len(t) > 65:
            # keyword가 포함된 위치 기준으로 앞뒤 문맥 추출
            idx = t.find(keyword)
            start = max(0, idx - 15)
            end = min(len(t), idx + len(keyword) + 40)
            t = ("…" if start > 0 else "") + t[start:end] + ("…" if end < len(t) else "")
        if t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


# ──────────────────────────────────────────────
# 5. UI — 상단 바
# ──────────────────────────────────────────────
gap = load_gap_lift()
quotes = extract_evidence_quotes()
keyword_freq = extract_keyword_frequencies()
dates, daily_series = daily_series_by_label()
stage_reaction = compute_stage_reaction()
positioning = load_positioning()

st.divider()

# ──────────────────────────────────────────────
# 6. 헤드라인 카드 2개
# ──────────────────────────────────────────────
weakest_label = min(EMPHASIS_LABELS, key=lambda l: gap["labels"][l]["lift"])
strongest_extra = "연주_악기"

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown(
            f"<span class='fsync-pill' style='background:{BAD_RED_BG}; color:{BAD_RED};'>"
            f"🔻 가장 아쉬운 포인트</span>",
            unsafe_allow_html=True,
        )
        lift_w = gap["labels"][weakest_label]["lift"]
        st.markdown(f"#### {LABEL_DISPLAY[weakest_label]} (×{lift_w})")
        st.caption("핵심 의도였지만 평소 수준에도 못 미쳤어요")
        if quotes[weakest_label]["top"]:
            top_quote, n = quotes[weakest_label]["top"][0]
            st.markdown(f"> _{top_quote}_ — {quotes[weakest_label]['n_hits']}건, 평소보다도 적었어요")

with col2:
    with st.container(border=True):
        st.markdown(
            f"<span class='fsync-pill' style='background:{GOOD_GREEN_BG}; color:{GOOD_GREEN};'>"
            f"✨ 기대 이상의 호응</span>",
            unsafe_allow_html=True,
        )
        lift_s = gap["labels"][strongest_extra]["lift"]
        st.markdown(f"#### {LABEL_DISPLAY[strongest_extra]} (×{lift_s})")
        st.caption("의도하지 않았는데 평소보다 더 화제됐어요")
        if quotes[strongest_extra]["top"]:
            top_quote, n = quotes[strongest_extra]["top"][0]
            st.markdown(f"> _{top_quote}_ — {quotes[strongest_extra]['n_hits']}건, 가장 많이 언급됨")

with st.container(border=True):
    st.markdown(
        f"<span class='fsync-pill' style='background:#FDF4E0; color:#A07A10;'>"
        f"💡 다음 컴백을 위한 제안</span>",
        unsafe_allow_html=True,
    )
    weak_all = [l for l in EMPHASIS_LABELS if gap["labels"][l]["verdict_clean"] == "누수"]
    weak_text = "·".join(LABEL_DISPLAY[l] for l in weak_all)
    st.markdown(
        f"{weak_text}은(는) 의도했지만 거의 평소 수준 그대로였어요. 무대 연출이나 인터뷰처럼 "
        f"직접적인 채널로 보강해보세요. 반대로 {LABEL_DISPLAY[strongest_extra]}는 의도 밖에서 "
        f"가장 화제됐으니 다음 콘텐츠에 적극 활용해볼 만해요."
    )

st.divider()

# ──────────────────────────────────────────────
# 7. 팬들이 많이 쓴 표현 — CSS 워드클라우드 (버튼 클릭 → 댓글)
# ──────────────────────────────────────────────
st.markdown("#### 팬들이 많이 쓴 표현")
st.caption("의도한 메시지 4가지 라벨 댓글에서 자주 나온 단어 · 크기 = 언급 댓글 수 · 단어를 클릭하면 관련 댓글 표시")

with st.container(border=True):
    # ── 키워드 수집: 9개 라벨 전체에서 뽑아 더 인사이트 있는 단어 확보 ──
    seen_kw = {}
    for label in ALL_9_LABELS:
        verdict = gap["labels"][label]["verdict_clean"] if label in gap["labels"] else "보통"
        for kw, cnt in keyword_freq[label]["top"][:10]:
            if kw not in seen_kw or cnt > seen_kw[kw][3]:
                seen_kw[kw] = (kw, label, verdict, cnt)

    raw_sorted = sorted(seen_kw.values(), key=lambda x: x[3], reverse=True)

    # 짧은 단어가 긴 단어의 접두사인 경우 긴 단어 제거
    def _redundant(kw, kept):
        for k in kept:
            if k != kw and (kw.startswith(k) or k.startswith(kw)):
                return True
        return False

    deduped, kept_kws = [], []
    for item in raw_sorted:
        kw = item[0]
        if not _redundant(kw, kept_kws):
            deduped.append(item)
            kept_kws.append(kw)
    # 빈도순 유지, 크기·소 혼합 배치를 위해 셔플 (대→소→소→대 인터리브)
    large  = [w for w in deduped if w[3] >= deduped[len(deduped)//3][3]]
    small  = [w for w in deduped if w[3] <  deduped[len(deduped)//3][3]]
    mixed = []
    li = si = 0
    while li < len(large) or si < len(small):
        if li < len(large): mixed.append(large[li]); li += 1
        if si < len(small): mixed.append(small[si]); si += 1
        if si < len(small): mixed.append(small[si]); si += 1
        if li < len(large): mixed.append(large[li]); li += 1
    word_options = mixed[:30]

    counts = [cnt for _, _, _, cnt in word_options]
    max_c  = max(counts) if counts else 1
    min_c  = min(counts) if counts else 1

    def _font(cnt):
        if max_c == min_c:
            return 26
        r = ((cnt - min_c) / (max_c - min_c)) ** 0.5
        return int(12 + r * 52)  # 12 ~ 64px

    # 수직 오프셋 — 인터리브 배치와 조합해 더 촘촘한 워드클라우드 느낌
    V_OFFSETS = [0, 14, -10, 20, -6, 16, -18, 8, -12, 22, 4, -8, 18, -14, 10,
                 -4, 24, -20, 6, -22, 12, -10, 18, -6, 14, -16, 8, -12, 20, 2]

    # ── CSS: 버튼을 투명 텍스트처럼, 컨테이너는 flex-wrap ──
    css_parts = []
    for i, (kw, label, verdict, cnt) in enumerate(word_options):
        c     = VERDICT_COLOR.get(verdict, VERDICT_COLOR["보통"])
        fsize = _font(cnt)
        voff  = V_OFFSETS[i % len(V_OFFSETS)]
        weight = 700 if fsize > 30 else 600
        css_parts.append(f"""
        .st-key-wc_cloud div[data-testid="stColumn"]:nth-of-type({i+1}) {{
            width: auto !important;
            min-width: 0 !important;
            flex: 0 0 auto !important;
            margin-top: {voff}px !important;
        }}
        .st-key-wc_cloud div[data-testid="stColumn"]:nth-of-type({i+1}) button {{
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 1px 4px !important;
            min-height: auto !important;
            height: auto !important;
        }}
        .st-key-wc_cloud div[data-testid="stColumn"]:nth-of-type({i+1}) button p {{
            font-size: {fsize}px !important;
            color: {c["line"]} !important;
            font-weight: {weight} !important;
            line-height: 1.1 !important;
            white-space: nowrap !important;
        }}
        .st-key-wc_cloud div[data-testid="stColumn"]:nth-of-type({i+1}) button:hover p {{
            text-decoration: underline !important;
            opacity: 0.75 !important;
        }}
        """)
    css_parts.append("""
    .st-key-wc_cloud div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 2px 0px !important;
        padding: 16px 12px !important;
        min-height: 280px;
    }
    """)
    st.markdown(f"<style>{''.join(css_parts)}</style>", unsafe_allow_html=True)

    with st.container(key="wc_cloud"):
        cols = st.columns(len(word_options))
        for i, (kw, label, verdict, cnt) in enumerate(word_options):
            with cols[i]:
                if st.button(kw, key=f"wc_btn_{i}"):
                    st.session_state["selected_word"] = (kw, label, verdict)

    # ── 범례 ──
    legend_cols = st.columns(3)
    for col, (v, lab) in zip(legend_cols, [
        ("정렬", "의도대로 통한 표현"),
        ("누수", "기대했지만 적었던 표현"),
        ("자생", "기대 이상의 호응 표현"),
    ]):
        with col:
            lc = VERDICT_COLOR[v]["line"]
            st.markdown(
                f"<div style='text-align:center;font-size:11px;'>"
                f"<span style='color:{lc};'>●</span> {lab}</div>",
                unsafe_allow_html=True,
            )

    # ── 선택 키워드 관련 댓글 ──
    if "selected_word" in st.session_state:
        sel_kw, sel_label, sel_verdict = st.session_state["selected_word"]
        c = VERDICT_COLOR[sel_verdict]
        st.markdown("---")
        st.markdown(
            f"<span style='background:{c['bg']};color:{c['text']};padding:4px 14px;"
            f"border-radius:999px;font-size:12px;font-weight:700;'>"
            f"「{sel_kw}」 · {LABEL_DISPLAY.get(sel_label, sel_label)} · {sel_verdict}</span>",
            unsafe_allow_html=True,
        )
        related = keyword_snippets_broad(sel_kw, limit=5)
        if related:
            for q in related:
                st.markdown(f"> _{q}_")
        else:
            st.caption("이 키워드를 포함한 댓글이 없어요")

st.divider()

# ──────────────────────────────────────────────
# 8. 라벨별 반응 추이 — 다중 선택 겹쳐보기 + 프로모션 마커
# ──────────────────────────────────────────────
st.markdown("#### 라벨별 반응 추이 & 컴백 단계별 비교")
st.caption("여러 라벨을 함께 선택하면 일별 추이와 단계별 비교를 동시에 볼 수 있어요")

if "selected_labels" not in st.session_state:
    st.session_state["selected_labels"] = set(EMPHASIS_LABELS)

label_order = EMPHASIS_LABELS + ["연주_악기", "장기_팬덤", "보컬_라이브", "신규_유입", "음악성", "이별_감성"]
cols = st.columns(len(label_order))
for i, label in enumerate(label_order):
    with cols[i]:
        is_selected = label in st.session_state["selected_labels"]
        if st.button(LABEL_DISPLAY[label], key=f"toggle_{label}",
                     type="primary" if is_selected else "secondary", use_container_width=True):
            if is_selected and len(st.session_state["selected_labels"]) > 1:
                st.session_state["selected_labels"].discard(label)
            else:
                st.session_state["selected_labels"].add(label)
            st.rerun()

selected = list(st.session_state["selected_labels"])

col_trend, col_stage = st.columns([1.3, 1])

with col_trend:
    st.markdown("##### 라벨별 반응 추이 · 프로모션 일정 포함")
    fig1, ax1 = plt.subplots(figsize=(7, 4.6))
    fig1.patch.set_facecolor("#FFFFFF")
    ax1.set_facecolor("#FFFFFF")
    x = pd.to_datetime(dates)
    y_max = max(max(daily_series[label]) for label in selected) if selected else 1

    for label in selected:
        lbl_text = LABEL_DISPLAY[label]
        if label in gap["labels"]:
            lbl_text += f" ×{gap['labels'][label]['lift']}"
        ax1.plot(x, daily_series[label], color=SERIES_COLOR[label], linewidth=2.4,
                  solid_capstyle="round", solid_joinstyle="round", label=lbl_text)

    COMEBACK_EVENT = "MV 발매(컴백)"
    promo_in_range = [ev for ev in PROMO_EVENTS if pd.Timestamp(ev["date"]) in x]
    others = [ev for ev in promo_in_range if ev["label"] != COMEBACK_EVENT]
    # 45도 대각선 텍스트 + 3단 높이 어긋남으로 날짜가 붙어 있어도 겹치지 않게 한다.
    # 컴백 이벤트는 가장 높은 전용 단을 혼자 차지해 다른 라벨과 절대 부딪히지 않게 분리한다.
    # 컨셉 포토(단체)·(개인)은 하루 차이로 붙어 있어 단을 강제로 분리한다.
    FORCE_HIGH_TIER = {"컨셉 포토(개인)"}
    tier_bases = [1.05, 1.25, 1.45]
    tier_cursor = 0
    for j, ev in enumerate(others):
        ev_date = pd.Timestamp(ev["date"])
        if ev["label"] in FORCE_HIGH_TIER:
            base = tier_bases[-1]
        else:
            base = tier_bases[tier_cursor % (len(tier_bases) - 1)]
            tier_cursor += 1
        ax1.axvline(ev_date, color="#C9A98A", linewidth=0.9, linestyle=":", alpha=0.9, zorder=1)
        ax1.scatter([ev_date], [y_max * (base - 0.02)], marker="v", s=18,
                    color="#C9A98A", zorder=4, clip_on=False)
        ax1.text(ev_date, y_max * base, ev["label"], rotation=45, fontsize=7.5,
                  color=SUGGEST_BROWN, ha="left", va="bottom",
                  rotation_mode="anchor", clip_on=False)

    comeback_date = pd.Timestamp(f"{[e for e in PROMO_EVENTS if e['label'] == COMEBACK_EVENT][0]['date']}")
    if comeback_date in x:
        cb_base = 1.55
        ax1.axvline(comeback_date, color=BRAND_RED, linewidth=1.5, alpha=0.9, zorder=2)
        ax1.scatter([comeback_date], [y_max * (cb_base - 0.02)], marker="v", s=40,
                    color=BRAND_RED, zorder=4, clip_on=False)
        ax1.text(comeback_date, y_max * cb_base, COMEBACK_EVENT, rotation=45, fontsize=8,
                  fontweight="bold", color=BRAND_RED, ha="left", va="bottom",
                  rotation_mode="anchor", clip_on=False)

    ax1.set_ylim(top=y_max * 2.3)
    ax1.grid(axis="y", color="#EFE7D8", linewidth=0.8)
    ax1.set_axisbelow(True)
    legend = ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                         ncol=min(3, max(len(selected), 1)), frameon=False)
    for spine in ax1.spines.values():
        spine.set_color(CARD_BORDER)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="x", rotation=60, labelsize=7, colors=SUGGEST_BROWN)
    ax1.tick_params(axis="y", labelsize=8, colors=SUGGEST_BROWN)
    fig1.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.34)
    st.pyplot(fig1)

with col_stage:
    st.markdown("##### 컴백 단계별 비교")
    stages_order = ["티저", "뮤직비디오", "음악방송", "부가 콘텐츠"]
    fig2, ax2 = plt.subplots(figsize=(6, 4.6))
    fig2.patch.set_facecolor("#FFFFFF")
    ax2.set_facecolor("#FFFFFF")
    n_labels = len(selected)
    width = 0.8 / max(n_labels, 1)
    x_pos = range(len(stages_order))
    for i, label in enumerate(selected):
        values = [stage_reaction[s][label] for s in stages_order]
        offset = (i - (n_labels - 1) / 2) * width
        ax2.bar([xi + offset for xi in x_pos], values, width, label=LABEL_DISPLAY[label],
                color=SERIES_COLOR[label], edgecolor="#FFFFFF", linewidth=0.6)
    ax2.grid(axis="y", color="#EFE7D8", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(stages_order, fontsize=10, color=SUGGEST_BROWN)
    ax2.set_ylabel("반응 비율(%)", color=SUGGEST_BROWN, fontsize=9)
    legend2 = ax2.legend(fontsize=8, frameon=True, facecolor="#FFFFFF", edgecolor=CARD_BORDER)
    legend2.get_frame().set_linewidth(1)
    for spine in ax2.spines.values():
        spine.set_color(CARD_BORDER)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8, colors=SUGGEST_BROWN)
    fig2.tight_layout()
    st.pyplot(fig2)

promo_text = " · ".join(f"{ev['date'][5:]} {ev['label']}" for ev in PROMO_EVENTS)
st.caption(f"프로모션 일정: {promo_text}")

st.divider()

# ──────────────────────────────────────────────
# 10. N.Flying 포지셔닝 · 밴드 아이돌 8팀 비교
# ──────────────────────────────────────────────
st.markdown("#### N.Flying 포지셔닝 · 밴드 아이돌 8팀 비교")
st.caption("9개 라벨 모두 다른 그룹과 비교해볼 수 있어요")

if "pos_label" not in st.session_state:
    st.session_state["pos_label"] = "밴드_정체성"

pos_cols = st.columns(len(ALL_9_LABELS))
for i, label in enumerate(ALL_9_LABELS):
    with pos_cols[i]:
        is_active = st.session_state["pos_label"] == label
        if st.button(LABEL_DISPLAY[label], key=f"pos_{label}",
                     type="primary" if is_active else "secondary", use_container_width=True):
            st.session_state["pos_label"] = label
            st.rerun()

pos_label = st.session_state["pos_label"]

with st.container(border=True):
    ranked = sorted(positioning.items(), key=lambda x: x[1][pos_label])  # 오름차순 → barh에서 위가 1위
    names = [a for a, _ in ranked]
    values = [p[pos_label] * 100 for _, p in ranked]

    fig3, ax3 = plt.subplots(figsize=(10, 4.2))
    fig3.patch.set_facecolor("#FFFFFF")
    ax3.set_facecolor("#FFFFFF")
    bar_colors = [BRAND_RED if a == "N.Flying" else "#E5DCC8" for a in names]
    ax3.barh(names, values, color=bar_colors, edgecolor="#FFFFFF", linewidth=0.6)
    ax3.grid(axis="x", color="#EFE7D8", linewidth=0.8)
    ax3.set_axisbelow(True)
    ax3.set_xlabel("반응 비율(%)", color=SUGGEST_BROWN)
    for spine in ax3.spines.values():
        spine.set_color(CARD_BORDER)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(colors=SUGGEST_BROWN)
    st.pyplot(fig3)

    nflying_rank = len(names) - names.index("N.Flying")  # names가 오름차순이므로 끝에서부터 순위
    if nflying_rank >= 5:
        st.error(f"N.Flying 순위: {nflying_rank} / {len(names)} — 동급 그룹 대비 약한 편이에요.")
    else:
        st.success(f"N.Flying 순위: {nflying_rank} / {len(names)} — 동급 그룹 대비 강한 편이에요.")
