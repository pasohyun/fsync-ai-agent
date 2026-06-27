import json, re, sys, os, torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model_training_script'))
import config
from model import RoBERTaMultiLabel
from transformers import AutoTokenizer

HANGUL_RE = re.compile(r"[가-힣]")
def is_korean(text, min_ratio=0.2):
    if not text or len(text.strip()) < 2: return False
    return len(HANGUL_RE.findall(text)) / len(text) >= min_ratio

SRC     = Path("환절기 데이터/comments_nflying_hwanjeolgi_test_before_excl_killingvoice_2026-06-26.jsonl")
OUT_DIR = Path("환절기 데이터/output")
OUT     = OUT_DIR / "comments_nflying_hwanjeolgi_test_before_excl_killingvoice_2026-06-26_me5_softmax.jsonl"

device     = torch.device('cuda:1')
model_name = 'intfloat/multilingual-e5-large'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = RoBERTaMultiLabel(model_name, config.NUM_LABELS).to(device)
model.load_state_dict(torch.load('model_output/me5_large_v2/best_model.pt', map_location=device))
model.eval()
print(f"모델 로드 완료 ({device})", flush=True)

raw      = []
with open(SRC, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: raw.append(json.loads(line))
        except: continue

filtered = [r for r in raw if is_korean(r.get('text',''))]
print(f"전체 {len(raw):,} → 한국어 {len(filtered):,}개\n", flush=True)

B = 64
with open(OUT, 'w', encoding='utf-8') as fout:
    for i in range(0, len(filtered), B):
        batch = [r['text'] for r in filtered[i:i+B]]
        enc   = tokenizer(batch, max_length=config.MAX_LEN, padding=True,
                          truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            logits  = model(enc['input_ids'], enc['attention_mask'])
            sigs    = torch.sigmoid(logits).cpu().tolist()
            softmax = torch.softmax(logits, dim=-1).cpu().tolist()

        for row, sig_list, sfx_list in zip(filtered[i:i+B], sigs, softmax):
            rec = {
                'comment_id':  row.get('comment_id'),
                'video_title': row.get('video_title'),
                'text':        row['text'],
                'sigmoid':     {l: round(v, 4) for l, v in zip(config.LABELS, sig_list)},
                'softmax':     {l: round(v, 4) for l, v in zip(config.LABELS, sfx_list)},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

        if i % (B * 10) == 0:
            print(f"  {min(i+B, len(filtered)):,} / {len(filtered):,}", flush=True)

print(f"\n✓ {OUT} 저장 완료", flush=True)
