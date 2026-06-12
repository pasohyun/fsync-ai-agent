import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

import config
from dataset import CommentDataset
from model import RoBERTaMultiLabel
from eval import evaluate


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
    os.makedirs(config.SAVE_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_dataset = CommentDataset(config.TRAIN_PATH, tokenizer, config.MAX_LEN)
    val_dataset = CommentDataset(config.VAL_PATH, tokenizer, config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    model = RoBERTaMultiLabel(config.MODEL_NAME, config.NUM_LABELS).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config.LR)

    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_micro_f1 = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        metrics = evaluate(model, tokenizer, config.VAL_PATH, config.THRESHOLD, device)

        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val micro_f1={metrics['micro_f1']:.4f} | "
            f"val macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = metrics["micro_f1"]
            save_path = os.path.join(config.SAVE_PATH, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  → best model 저장 (micro_f1={best_micro_f1:.4f})")

    print(f"\n학습 완료. best val micro_f1: {best_micro_f1:.4f}")


if __name__ == "__main__":
    main()
