import json
import os
import re
import sys
import argparse
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_training_script"))
import config
from model import RoBERTaMultiLabel

# OCR 사이드바 잡음 제거
NOISE_PATTERNS = [
    r"영화를 추월한 10조원의.*?우먼센스",
    r"편수로 보면 더욱 압도적이다.*?다르다\.",
    r"6월 기준 중국 내 숏폼.*?500억 회를 넘어선다\.",
    r"={3,}\s*PAGE\s*\d+\s*={3,}",
    r"\[출처\].*",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.DOTALL)


def clean_doc(text):
    text = NOISE_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def get_probs(text, model, tokenizer, device):
    enc = tokenizer(text, max_length=config.MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
    return {label: round(p, 4) for label, p in zip(config.LABELS, probs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model-dir", default="model_output/me5_large_v2")
    p.add_argument("--model-name", default="intfloat/multilingual-e5-large")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RoBERTaMultiLabel(args.model_name, config.NUM_LABELS).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.model_dir, "best_model.pt"), map_location=device))
    model.eval()
    print(f"모델 로드 완료: {args.model_name} ({device})")

    with open(args.input, encoding="utf-8") as f:
        raw = f.read()

    text = clean_doc(raw)
    print(f"텍스트 길이: {len(text)}자 (토크나이저 최대 {config.MAX_LEN} 토큰 → 초과 시 앞부분 사용)\n")

    probs = get_probs(text, model, tokenizer, device)

    print(f"{'라벨':20s} {'확률':>8}")
    print("─" * 32)
    for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"{label:20s} {prob:>8.4f}  {bar}")

    stem = re.sub(r"[^\w가-힣]", "_", os.path.splitext(os.path.basename(args.input))[0])[:40]
    out_path = args.out or os.path.join("gap_output", f"{stem}_txt_probs.json")
    os.makedirs("gap_output", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": args.input,
            "text_length": len(text),
            "text_preview": text[:300],
            "probs": probs,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {out_path} 저장")


if __name__ == "__main__":
    main()
