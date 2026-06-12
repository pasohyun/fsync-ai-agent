#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""게이트(Claude-사람 일치율): 골드 600개를 Claude로 blind 라벨 → 사람 라벨과 비교.

이것이 '본 시스템'(= bulk 라벨러)과 동일한 설정으로 라벨 품질을 재는 게이트다.
- 모델: Sonnet 4.6 (팀이 정한 bulk 라벨러. 최고 품질 보려면 MODEL을 claude-opus-4-8로)
- structured output(json_schema)으로 라벨을 10종 enum으로 강제 → 파싱 에러 0
- 프롬프트 캐싱: 라벨 가이드(system)를 배치 간 캐시
- blind: Claude엔 text만 보냄(사람 라벨 안 보여줌)

사전 준비:
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-...      # console.anthropic.com 에서 $10 충전 후 키 발급
실행:
  python3 src/labeling/claude_label_gold_600.py
"""
import json
import os
import sys
from collections import Counter

try:
    import anthropic
except ImportError:
    sys.exit("anthropic SDK 필요:  pip install anthropic")

# ── 설정 ──────────────────────────────────────────────────────────
GOLD = "data/train/band/gold/gold_labeled.jsonl"
OUT_LABELED = "data/train/band/gold/gold_claude_labeled.jsonl"
OUT_REPORT = "data/train/band/gold/gate_report.txt"
MODEL = "claude-sonnet-4-6"   # bulk 라벨러와 동일. (Opus 최고품질: "claude-opus-4-8")
BATCH_SIZE = 25
MAX_TOKENS = 8000

LABELS = ["밴드_에너지", "연주_악기", "보컬_라이브", "이별_감성", "위로_공감",
          "청량_여름", "신규_유입", "장기_팬덤", "역주행_기대", "기타_노이즈"]

GUIDE = """당신은 K-pop 유튜브 댓글을 분석하는 라벨링 전문가입니다.
각 댓글의 의미에 해당하는 라벨을 모두 부여하세요(멀티라벨).

원칙:
1. 멀티라벨: 한 댓글에 여러 라벨 가능. 해당하는 것 모두.
2. 표면 단어가 아니라 의미로 판단. (예: "0:56 극락" → 그 장면 몰입)
3. 억지로 끼우지 말 것: 어디에도 안 맞으면 기타_노이즈 하나만.

라벨 정의 (10종):
- 밴드_에너지: 밴드 정체성·락 사운드·돌파 에너지
- 연주_악기: 악기·연주(드럼/베이스/기타/세션)
- 보컬_라이브: 가창력·음색·라이브 보컬
- 이별_감성: 이별·슬픔·먹먹한 정서
- 위로_공감: 위로·공감·내 얘기 같다
- 청량_여름: 밝고 시원한·청량
- 신규_유입: 새로 유입(드라마·알고리즘·처음 알게 됨)
- 장기_팬덤: 오래 봐온 팬
- 역주행_기대: 역주행·숨은 명곡
- 기타_노이즈: 위 어디에도 안 맞는 응원·잡담·이모지·단순 좋아요"""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"results": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "integer"},
            "labels": {"type": "array", "items": {"type": "string", "enum": LABELS}},
        },
        "required": ["id", "labels"],
    }}},
    "required": ["results"],
}


def load_gold():
    rows = []
    with open(GOLD, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({"id": i, "text": str(d.get("text", "")),
                         "human": list(d.get("label", []))})
    return rows


def label_batch(client, batch):
    listing = "\n".join("ID=%d: %s" % (r["id"], r["text"].replace("\n", " ")) for r in batch)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": GUIDE, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content":
                   "다음 댓글들을 각각 라벨링하세요. 각 결과의 id는 입력의 ID와 정확히 일치해야 합니다.\n\n" + listing}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next(b.text for b in msg.content if b.type == "text")
    data = json.loads(text)
    out = {r["id"]: r["labels"] for r in data["results"]}
    return out, msg.usage


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    rows = load_gold()
    client = anthropic.Anthropic()
    print("골드 %d개 → Claude(%s) blind 라벨링 시작 (배치 %d)…" % (len(rows), MODEL, BATCH_SIZE))

    claude = {}
    in_tok = out_tok = cache_read = 0
    fails = []
    for s in range(0, len(rows), BATCH_SIZE):
        batch = rows[s:s + BATCH_SIZE]
        try:
            got, usage = label_batch(client, batch)
            claude.update(got)
            in_tok += usage.input_tokens
            out_tok += usage.output_tokens
            cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
            print("  %d/%d" % (min(s + BATCH_SIZE, len(rows)), len(rows)), end="\r", flush=True)
        except Exception as e:
            fails.append((s, str(e)[:120]))
            print("\n  ⚠ 배치 %d 실패: %s" % (s, str(e)[:120]))
    print()

    # 라벨 저장
    with open(OUT_LABELED, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps({"id": r["id"], "text": r["text"], "human": r["human"],
                                "claude": claude.get(r["id"], None)}, ensure_ascii=False) + "\n")

    # ── 일치율 계산 (Claude vs 사람) ───────────────────────────────
    scored = [r for r in rows if claude.get(r["id"]) is not None]
    n = len(scored)
    per = {lab: [0, 0, 0] for lab in LABELS}   # tp, fp, fn
    exact = 0
    jacc_sum = 0.0
    for r in scored:
        h, c = set(r["human"]), set(claude[r["id"]])
        if h == c:
            exact += 1
        u = h | c
        jacc_sum += (len(h & c) / len(u)) if u else 1.0
        for lab in LABELS:
            if lab in c and lab in h: per[lab][0] += 1
            elif lab in c and lab not in h: per[lab][1] += 1
            elif lab not in c and lab in h: per[lab][2] += 1

    lines = []
    def out(s=""): lines.append(s); print(s)
    out("=" * 60)
    out(" 게이트 리포트 — Claude(%s) vs 사람 골드" % MODEL)
    out("=" * 60)
    out(" 채점 대상: %d/%d  (실패 배치 %d)" % (n, len(rows), len(fails)))
    out("")
    out(" 라벨           P      R      F1    (Claude수/사람수)")
    macro = []
    TP = FP = FN = 0
    for lab in LABELS:
        tp, fp, fn = per[lab]
        p, rc, f = prf(tp, fp, fn)
        macro.append(f); TP += tp; FP += fp; FN += fn
        out("  %-10s %.2f   %.2f   %.2f    (%d/%d)" % (lab, p, rc, f, tp + fp, tp + fn))
    _, _, micro = prf(TP, FP, FN)
    out("")
    out(" micro-F1 = %.3f   macro-F1 = %.3f" % (micro, sum(macro) / len(macro)))
    out(" exact-match = %.1f%%   Jaccard = %.3f" % (100 * exact / n if n else 0, jacc_sum / n if n else 0))
    out("")
    out(" ※ micro/macro-F1 = '학생 모델(RoBERTa) 성능 상한선'. 이게 핵심 게이트 지표.")
    out(" ※ 사람 골드 자체 라벨러 편차(지민 멀티0 vs 소현 멀티33)가 있어 일치율에 영향.")
    # 비용
    cost = (in_tok / 1e6) * 3 + (out_tok / 1e6) * 15   # Sonnet $3/$15
    out("")
    out(" 토큰: in=%d (cache_read=%d) out=%d  →  대략 $%.3f (Sonnet 기준)" % (in_tok, cache_read, out_tok, cost))
    out("=" * 60)
    out(" 라벨 결과: %s" % OUT_LABELED)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(" 리포트 저장: %s" % OUT_REPORT)


if __name__ == "__main__":
    main()
