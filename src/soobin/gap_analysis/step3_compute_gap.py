import json
import os
import argparse
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "gap_output")

MUSIC_CONCEPT_LABELS = [
    "보컬_라이브", "연주_악기", "이별_감성", "청량_여름",
    "음악성", "밴드_정체성", "위로_공감"
]
FAN_BEHAVIOR_LABELS = [
    "장기_팬덤", "신규_유입", "역주행_기대", "비주얼_멤버매력", "기타_노이즈"
]


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--release", required=True, help="옥탑방/negane/everlasting")
    p.add_argument("--reaction", type=str, default=None,
                   help="gap_output/의 reaction json 파일명 (미지정시 자동탐색)")
    return p.parse_args()


def main():
    args = parse_args()

    intent_path = os.path.join(OUT_DIR, f"{args.release}_intent.json")
    if not os.path.exists(intent_path):
        raise FileNotFoundError(f"{intent_path} 없음 — step1 먼저 실행하세요")

    if args.reaction:
        reaction_path = os.path.join(OUT_DIR, args.reaction)
    else:
        candidates = [f for f in os.listdir(OUT_DIR) if f.startswith("reaction_")]
        if not candidates:
            raise FileNotFoundError("reaction json 없음 — step2 먼저 실행하세요")
        reaction_path = os.path.join(OUT_DIR, sorted(candidates)[-1])
        print(f"reaction 파일 자동 선택: {os.path.basename(reaction_path)}")

    with open(intent_path, encoding="utf-8") as f:
        intent_data = json.load(f)
    with open(reaction_path, encoding="utf-8") as f:
        reaction_data = json.load(f)

    intent = intent_data["intent_scores"]
    reaction = reaction_data["reaction_scores"]

    # 음악 컨셉 라벨 gap
    music_gap = {}
    for lbl in MUSIC_CONCEPT_LABELS:
        i_score = intent.get(lbl, 0.0)
        r_score = reaction.get(lbl, 0.0)
        music_gap[lbl] = {
            "intent": round(i_score, 3),
            "reaction": round(r_score, 3),
            "gap": round(r_score - i_score, 3),
        }

    # 코사인 유사도 (음악 컨셉 라벨만)
    i_vec = [intent.get(l, 0.0) for l in MUSIC_CONCEPT_LABELS]
    r_vec = [reaction.get(l, 0.0) for l in MUSIC_CONCEPT_LABELS]
    alignment = cosine(i_vec, r_vec)

    # 팬 행동 라벨 (반응만)
    fan_behavior = {lbl: round(reaction.get(lbl, 0.0), 3) for lbl in FAN_BEHAVIOR_LABELS}

    report = {
        "release": args.release,
        "reaction_source": os.path.basename(reaction_path),
        "n_comments": reaction_data["n_comments"],
        "cosine_alignment": round(alignment, 4),
        "music_concept_gap": music_gap,
        "fan_behavior_reaction": fan_behavior,
    }

    out_path = os.path.join(OUT_DIR, f"{args.release}_gap_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 출력
    print(f"\n{'='*55}")
    print(f"  Gap Report — {args.release}  (댓글 {reaction_data['n_comments']}개)")
    print(f"{'='*55}")
    print(f"  전체 일치율 (코사인): {alignment:.4f}")
    print(f"\n  [음악 컨셉 라벨 Gap]  (+기획초과 / -기획미달)")
    print(f"  {'라벨':18s} {'기획의도':>8} {'팬반응':>8} {'Gap':>8}")
    print(f"  {'-'*46}")
    for lbl, v in sorted(music_gap.items(), key=lambda x: x[1]["gap"]):
        arrow = "▲" if v["gap"] > 0.05 else ("▼" if v["gap"] < -0.05 else "─")
        print(f"  {lbl:18s} {v['intent']:>8.3f} {v['reaction']:>8.3f} {v['gap']:>+8.3f} {arrow}")
    print(f"\n  [팬 행동 라벨 — 자발적 반응]")
    for lbl, score in sorted(fan_behavior.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"  {lbl:18s} {score:.3f} {bar}")
    print(f"\n✓ {args.release}_gap_report.json 저장")


if __name__ == "__main__":
    main()
