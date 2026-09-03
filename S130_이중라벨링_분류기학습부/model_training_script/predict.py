import os
import torch
from transformers import AutoTokenizer

import config
from model import RoBERTaMultiLabel


def load_model(model_path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = os.path.join(config.SAVE_PATH, "best_model.pt")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = RoBERTaMultiLabel(config.MODEL_NAME, config.NUM_LABELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokenizer, device


def predict(text: str, model, tokenizer, threshold=config.THRESHOLD, device=None) -> dict:
    if device is None:
        device = next(model.parameters()).device

    encoding = tokenizer(
        text,
        max_length=config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    result = {label: round(prob, 4) for label, prob in zip(config.LABELS, probs)}
    predicted = [label for label, prob in result.items() if prob >= threshold]
    return {"probs": result, "predicted_labels": predicted}


if __name__ == "__main__":
    model, tokenizer, device = load_model()
    print("모델 로드 완료. 댓글을 입력하세요 (종료: q)\n")

    while True:
        text = input("댓글: ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue

        result = predict(text, model, tokenizer, device=device)
        print(f"예측 라벨: {result['predicted_labels']}")
        print("확률:")
        for label, prob in result["probs"].items():
            bar = "█" * int(prob * 20)
            print(f"  {label:6s} {prob:.4f} {bar}")
        print()
