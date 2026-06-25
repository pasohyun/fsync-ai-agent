import json
import os
import sys
import argparse
import random
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_training_script"))
import config

CORPUS = os.path.join(os.path.dirname(__file__), "..", "corpus_labeled_1.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "gap_output")


def build_from_labels(artist, n_samples, seed):
    random.seed(seed)
    with open(CORPUS, encoding="utf-8") as f:
        all_rows = [json.loads(l) for l in f if l.strip()]

    pool = [r for r in all_rows if r.get("artist") == artist] if artist else all_rows
    if len(pool) < n_samples:
        print(f"경고: {artist} 댓글이 {len(pool)}개뿐 (요청 {n_samples}개)")
        samples = pool
    else:
        samples = random.sample(pool, n_samples)

    label_count = defaultdict(int)
    for row in samples:
        for lbl in row.get("label", row.get("labels", [])):
            label_count[lbl] += 1

    n = len(samples)
    reaction_scores = {lbl: label_count.get(lbl, 0) / n for lbl in config.LABELS}

    return reaction_scores, n, samples


def build_from_model(samples, model_dir, gpu):
    import torch
    from transformers import AutoTokenizer
    from model import RoBERTaMultiLabel
    from torch.utils.data import DataLoader, Dataset

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = RoBERTaMultiLabel(config.MODEL_NAME, config.NUM_LABELS).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device))
    model.eval()

    texts = [r["text"] for r in samples]

    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            enc = tokenizer(self.texts[idx], max_length=config.MAX_LEN,
                            padding="max_length", truncation=True, return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0)}

    loader = DataLoader(TextDataset(texts), batch_size=config.BATCH_SIZE)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    import numpy as np
    mean_probs = np.mean(all_probs, axis=0)
    return {lbl: float(mean_probs[i]) for i, lbl in enumerate(config.LABELS)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artist", type=str, default="엔플라잉")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method", choices=["label", "model"], default="label",
                   help="label=기존 라벨 빈도, model=ME5-large v2 재추론")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    reaction_scores, n, samples = build_from_labels(args.artist, args.n, args.seed)

    if args.method == "model":
        if not args.model_dir:
            raise ValueError("--model-dir 필요")
        print(f"모델 추론 중... ({n}개 댓글)")
        reaction_scores = build_from_model(samples, args.model_dir, args.gpu)

    out_name = args.out_name or f"reaction_{args.artist}_{args.method}_{n}.json"
    out_path = os.path.join(OUT_DIR, out_name)
    out = {
        "artist": args.artist,
        "n_comments": n,
        "method": args.method,
        "reaction_scores": reaction_scores,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {out_name} 저장 ({n}개 댓글, method={args.method})")
    print("\n[라벨별 반응 비율]")
    for lbl, score in sorted(reaction_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"  {lbl:18s} {score:.3f} {bar}")


if __name__ == "__main__":
    main()
