import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report
import numpy as np

import config
from dataset import CommentDataset
from model import RoBERTaMultiLabel


def evaluate(model, tokenizer, val_path, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CommentDataset(val_path, tokenizer, config.MAX_LEN)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_label = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_label": {label: float(per_label[i]) for i, label in enumerate(config.LABELS)},
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = RoBERTaMultiLabel(config.MODEL_NAME, config.NUM_LABELS).to(device)

    model_path = os.path.join(config.SAVE_PATH, "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"모델 로드: {model_path}")

    metrics = evaluate(model, tokenizer, config.VAL_PATH, config.THRESHOLD, device)
    print(f"\n[Val 평가 결과]")
    print(f"  micro F1: {metrics['micro_f1']:.4f}")
    print(f"  macro F1: {metrics['macro_f1']:.4f}")
    print(f"\n[라벨별 F1]")
    for label, f1 in metrics["per_label"].items():
        print(f"  {label}: {f1:.4f}")


if __name__ == "__main__":
    main()
