import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

import config
from dataset import CommentDataset
from model import RoBERTaMultiLabel
from eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="사용할 GPU 인덱스")
    parser.add_argument("--model", type=str, default=None, help="모델명 (config 오버라이드)")
    parser.add_argument("--save-dir", type=str, default=None, help="저장 경로 (config 오버라이드)")
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기 (config 오버라이드)")
    return parser.parse_args()


def compute_sample_weights(label_vectors):
    label_matrix = np.array(label_vectors, dtype=np.float32)
    label_freq = label_matrix.sum(axis=0)
    label_freq = np.where(label_freq == 0, 1.0, label_freq)
    inv_freq = 1.0 / label_freq
    weights = (label_matrix * inv_freq).sum(axis=1)
    weights = np.where(weights == 0, inv_freq.min(), weights)
    return weights


def compute_pos_weights(label_vectors, device):
    label_matrix = np.array(label_vectors, dtype=np.float32)
    n_pos = label_matrix.sum(axis=0)
    n_neg = len(label_matrix) - n_pos
    pos_weight = n_neg / np.where(n_pos == 0, 1.0, n_pos)
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    args = parse_args()

    model_name = args.model or config.MODEL_NAME
    save_path = args.save_dir or config.SAVE_PATH
    batch_size = args.batch_size or config.BATCH_SIZE

    os.makedirs(save_path, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | model: {model_name} | batch: {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = CommentDataset(config.TRAIN_PATH, tokenizer, config.MAX_LEN)

    sample_weights = compute_sample_weights(train_dataset.label_vectors)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    model = RoBERTaMultiLabel(model_name, config.NUM_LABELS).to(device)
    pos_weight = compute_pos_weights(train_dataset.label_vectors, device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=config.LR)

    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_micro_f1 = 0.0
    history = []
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        metrics = evaluate(model, tokenizer, config.VAL_PATH, config.THRESHOLD, device, batch_size)

        is_best = metrics["micro_f1"] > best_micro_f1
        if is_best:
            best_micro_f1 = metrics["micro_f1"]
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))

        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val micro_f1={metrics['micro_f1']:.4f} | "
            f"val macro_f1={metrics['macro_f1']:.4f}"
            + (f"\n  → best model 저장 (micro_f1={best_micro_f1:.4f})" if is_best else "")
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_micro_f1": round(metrics["micro_f1"], 4),
            "val_macro_f1": round(metrics["macro_f1"], 4),
            "is_best": is_best,
        })

    log = {"model": model_name, "best_micro_f1": round(best_micro_f1, 4), "epochs": history}
    log_path = os.path.join(save_path, "train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n학습 완료. best val micro_f1: {best_micro_f1:.4f}")
    print(f"학습 로그 저장: {log_path}")


if __name__ == "__main__":
    main()
