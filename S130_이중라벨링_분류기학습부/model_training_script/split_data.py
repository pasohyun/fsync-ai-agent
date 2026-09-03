import json
import random

import config


def split_jsonl(src, train_out, val_out, train_ratio=0.7, seed=42):
    with open(src, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    for path, subset in [(train_out, train_data), (val_out, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"전체: {len(data)}개")
    print(f"train: {len(train_data)}개 → {train_out}")
    print(f"val  : {len(val_data)}개  → {val_out}")


if __name__ == "__main__":
    split_jsonl(config.LABELED_PATH, config.TRAIN_PATH, config.VAL_PATH)
