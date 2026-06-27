"""
댓글 JSONL → 한국어 필터 → ME5 v2 softmax 출력
Usage: python gap_analysis/run_softmax_comments.py --input <path.jsonl>
"""
import json, os, re, sys, argparse, torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_training_script"))
import config
from model import RoBERTaMultiLabel
from transformers import AutoTokenizer

HANGUL_RE = re.compile(r"[가-힣]")

def is_korean(text, min_ratio=0.2):
    if not text or len(text.strip()) < 2:
        return False
    return len(HANGUL_RE.findall(text)) / len(text) >= min_ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--model-dir', default='model_output/me5_large_v2')
    p.add_argument('--model-name', default='intfloat/multilingual-e5-large')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch', type=int, default=64)
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir  = in_path.parent
    stem     = in_path.stem
    out_path = out_dir / f"{stem}_me5_softmax.jsonl"

    # ── 모델 로드 ──────────────────────────────────────────
    device    = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = RoBERTaMultiLabel(args.model_name, config.NUM_LABELS).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.model_dir, 'best_model.pt'), map_location=device))
    model.eval()
    print(f"모델 로드 완료 ({device})\n라벨: {config.LABELS}\n")

    # ── 데이터 로드 & 전처리 ───────────────────────────────
    raw = [json.loads(l) for l in in_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    total = len(raw)

    filtered = [r for r in raw if is_korean(r.get('text', ''))]
    print(f"전체: {total:,}  →  한국어 필터 후: {len(filtered):,}  ({total - len(filtered):,}개 제거)\n")

    # ── 배치 추론 ──────────────────────────────────────────
    results = []
    texts   = [r['text'] for r in filtered]
    B       = args.batch

    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        enc = tokenizer(batch, max_length=config.MAX_LEN, padding=True,
                        truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(enc['input_ids'], enc['attention_mask'])
            probs  = torch.softmax(logits, dim=-1).cpu().tolist()

        for row, prob_list in zip(filtered[i:i+B], probs):
            scores = {l: round(v, 4) for l, v in zip(config.LABELS, prob_list)}
            top    = max(scores, key=scores.get)
            results.append({
                'comment_id':   row.get('comment_id'),
                'video_title':  row.get('video_title'),
                'text':         row['text'],
                'top_label':    top,
                'softmax':      scores,
            })

        if (i // B) % 10 == 0:
            print(f"  {min(i+B, len(texts)):,} / {len(texts):,} 처리중...")

    # ── 저장 ──────────────────────────────────────────────
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n✓ {out_path} 저장 완료 ({len(results):,}개)")

    # ── 라벨 분포 요약 ─────────────────────────────────────
    from collections import Counter
    top_counts = Counter(r['top_label'] for r in results)
    print(f"\n[ top_label 분포 ]")
    for l, cnt in top_counts.most_common():
        bar = '█' * int(cnt / len(results) * 40)
        print(f"  {l:<14} {cnt:>5}개  {cnt/len(results)*100:5.1f}%  {bar}")


if __name__ == '__main__':
    main()
