#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10만 통합 코퍼스 → Claude Batch API로 11라벨 멀티라벨 라벨링.

- 모델: Opus 4.8 (품질). Batch API = 50% 할인.
- 11라벨 enum + few-shot 가이드(정의+O/X 예시+멀티라벨 정책) structured output 강제 → 파싱 에러 0.
- 댓글 25개씩 1요청으로 묶음(108k → 약 4,300요청, Batch 한도 10만 이내).
- 골드 600(사람 검증 holdout)은 학습 누수 방지 위해 제외.
- 중단돼도 batch_id 파일로 재개(재실행하면 제출 건너뛰고 수집부터).

사전 준비:
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-...     # 셸 프로필(~/.zshrc)에. 채팅/깃에 넣지 말 것.
사용:
  python3 src/labeling/label_corpus_batch.py --limit 50     # 스모크(50개, 몇 센트) — 본런 전 점검
  python3 src/labeling/label_corpus_batch.py                # 전수(108k)
  python3 src/labeling/label_corpus_batch.py --model claude-sonnet-4-6   # 모델 바꾸려면
"""
import argparse
import json
import os
import sys
import time

try:
    import anthropic
except ImportError:
    sys.exit("anthropic SDK 필요:  pip install anthropic")

# ── 경로/설정 ─────────────────────────────────────────────────────
CORPUS = "data/train/band/band_korean_all.jsonl"
GOLD = "data/train/band/gold/gold_labeled.jsonl"          # 제외용(holdout)
OUT = "data/train/band/corpus_labeled.jsonl"
BATCH_ID_FILE = "data/train/band/.batch_id"               # 재개용
MODEL = "claude-opus-4-8"
PER_REQ = 25            # 요청당 댓글 수
MAX_TOKENS = 4000
POLL_SEC = 60

LABELS = ["밴드_정체성", "연주_악기", "보컬_라이브", "비주얼_멤버매력", "이별_감성",
          "위로_공감", "청량_여름", "신규_유입", "장기_팬덤", "역주행_기대", "기타_노이즈"]

# ── few-shot 가이드 (팀 검토 반영 최종본) ─────────────────────────
GUIDE = """당신은 K-pop 밴드 유튜브 댓글 라벨러입니다. 각 댓글에 의미가 명확히 드러나는 라벨을 모두 부여하세요(멀티라벨). 표면 단어가 아니라 실제 의미로 판단합니다.

[라벨 11종 — 정의 + O(맞음)/X(헷갈리지만 다른 라벨)]
1. 밴드_정체성 — 밴드라는 형식·정체성·락 색깔, 또는 합주/락 사운드 에너지에 대한 인식·호감.
   O "이게 락밴드라는거야" / "네명이서 밴드해줘서 고맙다" / "이것이 밴드의 맛"
   X "기타 짱"(악기→연주_악기) / "밴드는 라이브가 짱"(가창→보컬_라이브) / "트렌디한 느낌"(→청량_여름) / "선배밴드처럼 흥했음"(흥행기원→역주행_기대)
2. 연주_악기 — 특정 악기·연주 디테일(드럼/베이스/기타솔로/리프/건반/세션)을 콕 집음.
   O "베이스라인 왤케 좋아" / "기타솔로ㄷㄷ" / "신시사이저 사운드 굿"
   X "드럼 치는 애 잘생겼다"(외모→비주얼_멤버매력) / "애들 연주 미쳤다"(특정 악기 없는 막연→밴드_정체성)
3. 보컬_라이브 — 보컬 실력·음색·고음·라이브 가창에 초점.
   O "유회승 음색 너무 좋다" / "라이브가 음원을 찢었다" / "고음 차력쇼 미쳤다"
   X "빨간머리가 보컬인가요?"(식별 질문→기타_노이즈) / "노래란 노래 다 잘부른다"(막연→기타_노이즈)
4. 비주얼_멤버매력 — 멤버 본인의 외모·스타일링·표정·눈빛·멤버 간 케미 호감.
   O "김재현 잘생겼다 눈빛에 타죽겠어" / "둘이 케미 너무 좋아" / "승짱 표정 치명적"
   X "최우식 연기 개지림"(게스트배우→기타_노이즈) / "뮤비 색감 예쁘다"(영상미→기타_노이즈)
5. 이별_감성 — 이별·상실을 자기 경험으로 이입한 회복 이전의 정적 슬픔·그리움·먹먹함.
   O "이 노래 알려준 전여친 생각나네" / "헤어지고 나니 재생 버튼도 못 누르겠네"
   X "치유받는 기분에 눈물"(회복→위로_공감) / "노래 슬프네"(막연→기타_노이즈)
6. 위로_공감 — 곡/가사가 '나를' 위로·치유·해소시켰다는 1인칭 수용 정서.
   O "오늘 힘든 일 있었는데 위로가 되네" / "노래로 위로받을 수 있구나"
   X "힘내세요 푹 쉬어요"(위로를 주는 발화→기타_노이즈) / "아련해지며 기분 좋아짐"(→이별_감성)
7. 청량_여름 — 명시적 청량/시원/산뜻 '그리고' 계절감(여름·바다·날씨)의 청량 무드. ※'시원'이라는 단어만으로는 부족 — 곡/뮤비의 계절·날씨 청량 무드여야 함.
   O "완전 여름재질 배경 속초 요트" / "포카리에서 수영하는 듯 청량"
   X "아따 시원시원하다"(계절감 없는 막연→기타_노이즈) / "목소리 시원시원"(음색→보컬_라이브) / "신나요"(경쾌한 사운드→밴드_정체성)
8. 신규_유입 — 본인이 새로/외부에서 유입됨을 진술(보고 왔다·추천·처음·알고리즘·커버 듣고·방금 입덕).
   O "음방에서 우연히 보고 계속 듣고있어요" / "알고리즘 일 잘하네" / "쇼츠로 발견해 뮤비까지 찾아옴"
   X "지금도 듣는 사람~?"(타인 호출→역주행_기대) / "2024년부터 듣기시작"(경로 없는 막연→기타_노이즈)
9. 장기_팬덤 — 데뷔 때부터/오래 함께한 기존 팬의 지속 애정·소속감·데뷔주년·컴백 축하·"오래 가자".
   O "데뷔10주년 축하 영원하자" / "데뷔무대부터 직캠 찾아봤다"
   X "지금도 듣는 사람~?"(현재 청취자 호출→역주행_기대)
   ※입덕 시점 규칙: '방금/오늘 입덕'=신규_유입 / '~로 입덕한 사람으로 (지금까지·컴백축하)'·'내 입덕곡'=장기_팬덤(과거에 입덕해 지금까지 팬).
10. 역주행_기대 — 숨은 명곡의 흥행·차트 역주행을 기원/예측하거나 스밍 독려("더 떠야 한다").
    O "엔플라잉 명곡들 다 역주행하길" / "노래 좋은데 좀 더 확 떠라"
    X "아 진짜 띵곡이다"(단독 감탄→기타_노이즈) / "왜 역주행하는거에요?"(현상 질문→신규_유입)
11. 기타_노이즈 — 위 10개 근거가 하나도 없을 때만(이모지·단순 좋아요·외국어 인사·게스트배우·외부맥락·대상 불특정 막연 칭찬). 항상 단독으로만.
    O "와우!!!!!" / "노래 좋네"(대상 불특정 막연 칭찬)
    X "기타 잘생겼다"(악기→연주_악기) / "엔플라잉 대박나라"(흥행기원→역주행_기대)

[멀티라벨 정책]
- 근거가 명확한 라벨만. 서로 다른 축을 동시에 또렷이 만족하면 복수 부여(예: "키보드 소리 예쁘고 승짱도 예쁘다"=연주_악기+비주얼_멤버매력).
- 애매하면 핵심 의미 1개만. 표면 단어·분위기만 보고 끼우는 억지 다중 라벨 금지(키워드 함정 주의).
- 멤버 단어·연도·'밴드/라이브/역주행/명곡' 키워드가 표면에 있어도 의미가 다르면 부여하지 않는다.
- 다른 10개 근거가 전혀 없을 때만 기타_노이즈, 기타_노이즈는 항상 단독(다른 라벨과 동시 금지)."""

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


def load_corpus(limit, gold_ids):
    rows = []
    for line in open(CORPUS, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        cid = str(d.get("comment_id", ""))
        if cid in gold_ids:                      # 골드(holdout) 제외
            continue
        rows.append({"comment_id": cid, "text": str(d.get("text", "")),
                     "artist": d.get("artist"), "video_type": d.get("video_type")})
        if limit and len(rows) >= limit:
            break
    return rows


def build_requests(rows):
    from anthropic.types.messages.batch_create_params import Request
    reqs = []
    for s in range(0, len(rows), PER_REQ):
        chunk = rows[s:s + PER_REQ]
        listing = "\n".join("ID=%d: %s" % (s + i, c["text"].replace("\n", " "))
                            for i, c in enumerate(chunk))
        reqs.append(Request(
            custom_id="req-%d" % s,
            params={
                "model": MODEL, "max_tokens": MAX_TOKENS,
                "system": [{"type": "text", "text": GUIDE, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content":
                              "다음 댓글들을 각각 라벨링하세요. 각 결과 id는 입력 ID와 정확히 일치.\n\n" + listing}],
                "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
            },
        ))
    return reqs


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만(스모크). 0=전수")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model

    client = anthropic.Anthropic()
    gold_ids = set()
    if os.path.exists(GOLD):
        for line in open(GOLD, encoding="utf-8"):
            try:
                gold_ids.add(str(json.loads(line).get("comment_id", "")))
            except Exception:
                pass
    rows = load_corpus(args.limit, gold_ids)
    print("대상 댓글 %d개 (골드 %d 제외), 모델 %s" % (len(rows), len(gold_ids), MODEL))

    # ── 1) 배치 제출(또는 재개) ──
    batch_id = None
    if os.path.exists(BATCH_ID_FILE):
        batch_id = open(BATCH_ID_FILE).read().strip()
        print("기존 배치 재개:", batch_id)
    else:
        reqs = build_requests(rows)
        print("요청 %d개(댓글 %d/요청) 제출 중…" % (len(reqs), PER_REQ))
        batch = client.messages.batches.create(requests=reqs)
        batch_id = batch.id
        open(BATCH_ID_FILE, "w").write(batch_id)
        print("배치 제출됨:", batch_id, "(대부분 1시간 내, 최대 24h)")

    # ── 2) 폴링 ──
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        rc = b.request_counts
        print("  상태 %s | 처리중 %d 완료 %d 실패 %d" %
              (b.processing_status, rc.processing, rc.succeeded, rc.errored), flush=True)
        time.sleep(POLL_SEC)

    # ── 3) 결과 수집 → comment_id 매핑 후 저장 ──
    labels_by_id = {}
    in_tok = out_tok = cache_read = fails = 0
    for res in client.messages.batches.results(batch_id):
        if res.result.type != "succeeded":
            fails += 1
            continue
        msg = res.result.message
        u = msg.usage
        in_tok += u.input_tokens; out_tok += u.output_tokens
        cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        try:
            text = next(b.text for b in msg.content if b.type == "text")
            for r in json.loads(text)["results"]:
                labels_by_id[int(r["id"])] = r["labels"]
        except Exception as e:
            fails += 1
            print("  ⚠ 파싱 실패 %s: %s" % (res.custom_id, str(e)[:80]))

    with open(OUT, "w", encoding="utf-8") as w:
        done = 0
        for i, c in enumerate(rows):
            labs = labels_by_id.get(i)
            if labs is None:
                continue
            done += 1
            w.write(json.dumps({"comment_id": c["comment_id"], "artist": c["artist"],
                                "video_type": c["video_type"], "text": c["text"],
                                "label": labs, "label_source": "claude-%s" % MODEL},
                               ensure_ascii=False) + "\n")

    # Opus $5/$25, Sonnet $3/$15 — Batch 50%↓
    rate = (2.5, 12.5) if "opus" in MODEL else (1.5, 7.5)
    cost = in_tok / 1e6 * rate[0] + out_tok / 1e6 * rate[1]
    from collections import Counter
    dist = Counter()
    for labs in labels_by_id.values():
        for l in labs:
            dist[l] += 1
    print("=" * 60)
    print(" 라벨링 완료: %d개 → %s (실패 %d)" % (done, OUT, fails))
    print(" 토큰 in=%d (cache_read=%d) out=%d → 약 $%.2f (Batch)" % (in_tok, cache_read, out_tok, cost))
    print(" 라벨 분포:")
    for l in LABELS:
        n = dist.get(l, 0)
        print("   %-12s %6d (%.1f%%)" % (l, n, 100 * n / max(done, 1)))
    print("=" * 60)
    print(" ※ 완료 후 .batch_id 파일 삭제해야 다음 런이 새로 제출됨:  rm", BATCH_ID_FILE)


if __name__ == "__main__":
    main()
