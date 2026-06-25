import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

import config
from dataset import CommentDataset
from model import RoBERTaMultiLabel


def evaluate(model, tokenizer, val_path, threshold=0.5, device=None, batch_size=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    dataset = CommentDataset(val_path, tokenizer, config.MAX_LEN)
    loader = DataLoader(dataset, batch_size=batch_size)

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

    micro_precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    pl_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    pl_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    pl_micro_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    pl_macro_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return {
        "micro_precision": micro_precision,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_label_precision": {label: float(pl_precision[i]) for i, label in enumerate(config.LABELS)},
        "per_label_recall": {label: float(pl_recall[i]) for i, label in enumerate(config.LABELS)},
        "per_label_micro_f1": {label: float(pl_micro_f1[i]) for i, label in enumerate(config.LABELS)},
        "per_label_macro_f1": {label: float(pl_macro_f1[i]) for i, label in enumerate(config.LABELS)},
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="사용할 GPU 인덱스")
    parser.add_argument("--model", type=str, default=None, help="모델명 (config 오버라이드)")
    parser.add_argument("--model-dir", type=str, default=None, help="모델 로드 경로")
    parser.add_argument("--save-dir", type=str, default=None, help="결과 저장 경로 (미지정시 model-dir과 동일)")
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기 (config 오버라이드)")
    parser.add_argument("--val-path", type=str, default=None, help="평가 데이터 경로 (config 오버라이드)")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model or config.MODEL_NAME
    model_dir = args.model_dir or args.save_dir or config.SAVE_PATH
    save_path = args.save_dir or model_dir
    batch_size = args.batch_size or config.BATCH_SIZE
    val_path = args.val_path or config.VAL_PATH

    os.makedirs(save_path, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RoBERTaMultiLabel(model_name, config.NUM_LABELS).to(device)

    model_path = os.path.join(model_dir, "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"모델 로드: {model_path}")

    metrics = evaluate(model, tokenizer, val_path, config.THRESHOLD, device, batch_size)
    print(f"\n[Val 평가 결과]")
    print(f"  {'':20s} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'micro':20s} {metrics['micro_precision']:>10.4f} {metrics['micro_recall']:>10.4f} {metrics['micro_f1']:>10.4f}")
    print(f"  {'macro':20s} {metrics['macro_precision']:>10.4f} {metrics['macro_recall']:>10.4f} {metrics['macro_f1']:>10.4f}")
    print(f"\n[라벨별]")
    print(f"  {'라벨':20s} {'Precision':>10} {'Recall':>10} {'micro F1':>10} {'macro F1':>10}")
    for label in config.LABELS:
        print(f"  {label:20s} {metrics['per_label_precision'][label]:>10.4f} {metrics['per_label_recall'][label]:>10.4f} {metrics['per_label_micro_f1'][label]:>10.4f} {metrics['per_label_macro_f1'][label]:>10.4f}")

    out = {"model": model_name, **metrics}
    out_path = os.path.join(save_path, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
