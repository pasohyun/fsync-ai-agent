import json
import os
import sys
import argparse
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_training_script"))
import config
from model import RoBERTaMultiLabel


def get_probs(text, model, tokenizer, device):
    enc = tokenizer(text, max_length=config.MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
    return {label: round(p, 4) for label, p in zip(config.LABELS, probs)}


def extract_texts(intent_json):
    texts = {}
    texts["core_intent"] = intent_json.get("core_intent", "").strip()

    profile = intent_json.get("intended_label_profile", {})
    for label, val in profile.items():
        if isinstance(val, dict) and val.get("evidence") and val["evidence"].get("quote"):
            texts[f"evidence_{label}"] = val["evidence"]["quote"].strip()

    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="intent_release/intent_vectors/intent_hwanjeolgi.json")
    p.add_argument("--model-dir", default="model_output/me5_large_v2")
    p.add_argument("--model-name", default="intfloat/multilingual-e5-large")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RoBERTaMultiLabel(args.model_name, config.NUM_LABELS).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.model_dir, "best_model.pt"), map_location=device))
    model.eval()
    print(f"모델 로드 완료: {args.model_name} ({device})\n")

    with open(args.input, encoding="utf-8") as f:
        intent_json = json.load(f)

    release = intent_json.get("release", "unknown")
    texts = extract_texts(intent_json)

    results = {}
    for name, text in texts.items():
        if not text:
            continue
        probs = get_probs(text, model, tokenizer, device)
        results[name] = probs

        print(f"[{name}]")
        print(f"  텍스트: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  {'라벨':20s} {'확률':>8}")
        for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"  {label:20s} {prob:>8.4f}  {bar}")
        print()

    out_path = f"gap_output/{release}_model_probs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"release": release, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"✓ {out_path} 저장")


if __name__ == "__main__":
    main()
