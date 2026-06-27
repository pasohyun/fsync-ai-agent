import json, re, sys, os, torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model_training_script'))
import config
from model import RoBERTaMultiLabel
from transformers import AutoTokenizer

HANGUL_RE = re.compile(r"[가-힣]")
def is_korean(text, min_ratio=0.2):
    if not text or len(text.strip()) < 2: return False
    return len(HANGUL_RE.findall(text)) / len(text) >= min_ratio

SRC     = Path("corpus_labeled_1.jsonl")
OUT_DIR = Path("전체 데이터/OUTPUT")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SIG = OUT_DIR / "corpus_me5_sigmoid.jsonl"

LABELS_8 = ['비주얼_멤버매력','보컬_라이브','밴드_정체성','위로_공감','연주_악기','이별_감성','청량_여름','음악성']
LABELS_7 = ['보컬_라이브','밴드_정체성','위로_공감','연주_악기','이별_감성','청량_여름','음악성']

# ── 모델 로드 ────────────────────────────────────────────────
device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'intfloat/multilingual-e5-large'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = RoBERTaMultiLabel(model_name, config.NUM_LABELS).to(device)
model.load_state_dict(torch.load('model_output/me5_large_v2/best_model.pt', map_location=device))
model.eval()
print(f"모델 로드 완료 ({device})", flush=True)

# ── 데이터 로드 ──────────────────────────────────────────────
raw = []
with open(SRC, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            raw.append(json.loads(line))
        except json.JSONDecodeError:
            continue

filtered = [r for r in raw if is_korean(r.get('text',''))]
print(f"전체 {len(raw):,} → 한국어 {len(filtered):,}개\n", flush=True)

# ── sigmoid 추론 ─────────────────────────────────────────────
B = 128
results = []
fout = open(OUT_SIG, 'w', encoding='utf-8')

for i in range(0, len(filtered), B):
    batch = [r['text'] for r in filtered[i:i+B]]
    enc   = tokenizer(batch, max_length=config.MAX_LEN, padding=True,
                      truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'])
        sigs   = torch.sigmoid(logits).cpu().tolist()
    for row, sig_list in zip(filtered[i:i+B], sigs):
        sigmoid = {l: round(v, 4) for l, v in zip(config.LABELS, sig_list)}
        rec = {'comment_id': row.get('comment_id'), 'artist': row.get('artist'),
               'video_type': row.get('video_type'), 'text': row['text'], 'sigmoid': sigmoid}
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        results.append(rec)
    if i % (B * 50) == 0:
        print(f"  {min(i+B, len(filtered)):,} / {len(filtered):,}", flush=True)

fout.close()
print(f"\n✓ {OUT_SIG} ({len(results):,}개)", flush=True)

# ── 리액션 파일 ──────────────────────────────────────────────
def save_thresh(rows, label_set, tag, thresh):
    hits, n_hit, n_multi = [], 0, 0
    for r in rows:
        matched = [l for l in label_set if r['sigmoid'].get(l,0) > thresh]
        if matched:
            hits.extend(matched); n_hit += 1
            if len(matched) >= 2: n_multi += 1
    total  = len(hits); counts = Counter(hits)
    scores = {l: round(counts.get(l,0)/total, 4) for l in label_set}
    pcts   = {l: round(counts.get(l,0)/total*100, 2) for l in label_set}
    result = {'method': f'sigmoid_threshold_{thresh}', 'threshold': thresh,
              'n_total': len(rows), 'n_comments_hit': n_hit, 'n_multi_hit': n_multi,
              'n_label_hits': total, 'labels': label_set,
              'reaction_pct': scores, 'reaction_pct_percent': pcts,
              'hit_counts': {l: counts.get(l,0) for l in label_set}}
    tstr = str(thresh).replace('.','')
    fname = OUT_DIR / f"corpus_reaction_{tstr[-2:]}_{tag}.json"
    fname.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    top3 = sorted(pcts.items(), key=lambda x:-x[1])[:3]
    print(f"✓ {fname.name}  히트:{n_hit:,}  멀티:{n_multi}  top3:{top3}", flush=True)

def save_top1(rows, label_set, tag):
    top1s  = [max(label_set, key=lambda l: r['sigmoid'].get(l,0)) for r in rows]
    n      = len(top1s); counts = Counter(top1s)
    scores = {l: round(counts.get(l,0)/n, 4) for l in label_set}
    pcts   = {l: round(counts.get(l,0)/n*100, 2) for l in label_set}
    result = {'method': 'sigmoid_top1', 'n_total': len(rows), 'labels': label_set,
              'reaction_pct': scores, 'reaction_pct_percent': pcts,
              'hit_counts': {l: counts.get(l,0) for l in label_set}}
    fname = OUT_DIR / f"corpus_reaction_top1_{tag}.json"
    fname.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    top3 = sorted(pcts.items(), key=lambda x:-x[1])[:3]
    print(f"✓ {fname.name}  top3:{top3}", flush=True)

for label_set, tag in [(LABELS_8,'8label'),(LABELS_7,'7label')]:
    save_thresh(results, label_set, tag, 0.80)
    save_thresh(results, label_set, tag, 0.85)
    save_top1(results, label_set, tag)

print("\n✓ 전체 완료", flush=True)
