import json
import os

LEVEL_SCORE = {"high": 1.0, "med": 0.67, "low": 0.33, "none": 0.0}

MUSIC_CONCEPT_LABELS = {
    "보컬_라이브", "연주_악기", "이별_감성", "청량_여름",
    "음악성", "밴드_정체성", "위로_공감"
}
FAN_BEHAVIOR_LABELS = {
    "장기_팬덤", "신규_유입", "역주행_기대", "비주얼_멤버매력", "기타_노이즈"
}

SRC = os.path.join(os.path.dirname(__file__), "..", "intent_release", "intent_12label", "intent_vectors_12.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "gap_output")

def build():
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)

    for profile in data["profiles"]:
        release = profile["release"]
        intent_scores = {}
        for entry in profile["profile"]:
            label = entry["label"]
            score = LEVEL_SCORE.get(entry["level"], 0.0)
            intent_scores[label] = score

        out = {
            "release": release,
            "intent_scores": intent_scores,
            "music_concept_labels": sorted(MUSIC_CONCEPT_LABELS),
            "fan_behavior_labels": sorted(FAN_BEHAVIOR_LABELS),
        }
        out_path = os.path.join(OUT_DIR, f"{release}_intent.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"✓ {release}_intent.json 저장")
        for label, score in sorted(intent_scores.items(), key=lambda x: -x[1]):
            tag = "[컨셉]" if label in MUSIC_CONCEPT_LABELS else "[행동]"
            print(f"   {tag} {label}: {score}")
        print()

if __name__ == "__main__":
    build()
