import json
import torch
from torch.utils.data import Dataset
from config import LABELS


def load_jsonl(path):
    texts, raw_labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item["text"])
            raw_labels.append(item.get("labels", []))
    return texts, raw_labels


def build_label_vectors(raw_labels):
    vectors = []
    for label_list in raw_labels:
        vec = [1.0 if l in label_list else 0.0 for l in LABELS]
        vectors.append(vec)
    return vectors


class CommentDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len):
        texts, raw_labels = load_jsonl(jsonl_path)
        self.texts = texts
        self.label_vectors = build_label_vectors(raw_labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_vectors[idx], dtype=torch.float),
        }
