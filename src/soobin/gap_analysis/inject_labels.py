"""
intent_release/02_intent_extraction/ 의 JSON 파일 각각에
keywords / lines 항목마다 top_label, top_score, second_label, second_score 추가.
keywords는 string → {"text": ..., "top_label": ..., ...} 으로 변환.
"""
import json, os, sys, torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_training_script"))
import config
from transformers import AutoTokenizer, AutoModel

SRC_DIR   = 'intent_release/02_intent_extraction/'
DESC_FILE = os.path.join(os.path.dirname(__file__), 'label_descriptions.json')
MODEL_NAME = 'intfloat/multilingual-e5-large'

NORM_MIN = 0.70
NORM_MAX = 0.92


def normalize(v):
    return round(float(np.clip((v - NORM_MIN) / (NORM_MAX - NORM_MIN), 0.0, 1.0)), 4)


def average_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1)


def embed(texts, tokenizer, model, device, prefix='query'):
    prefixed = [f"{prefix}: {t}" for t in texts]
    enc = tokenizer(prefixed, max_length=512, padding=True,
                    truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**enc)
    emb = average_pool(out.last_hidden_state, enc['attention_mask'])
    return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()


def top2(text_emb, label_kw_embs, active_labels):
    scores = {}
    for l, kw_embs in zip(active_labels, label_kw_embs):
        sims = np.dot(kw_embs, text_emb)
        scores[l] = normalize(sims.max())
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[0][0], ranked[0][1], ranked[1][0], ranked[1][1]


def main():
    with open(DESC_FILE, encoding='utf-8') as f:
        label_desc_map = json.load(f)

    active_labels = [l for l in config.LABELS if label_desc_map.get(l, '').strip()]
    print(f"활성 라벨 ({len(active_labels)}개): {active_labels}\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_enc = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model_enc.eval()
    print(f"ME5 로드 완료 ({device})\n{'='*60}")

    label_kw_embs = []
    for l in active_labels:
        kws  = [k.strip() for k in label_desc_map[l].split(',') if k.strip()]
        embs = embed(kws, tokenizer, model_enc, device, prefix='passage')
        label_kw_embs.append(embs)

    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(SRC_DIR, fname)
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)

        album    = data.get('album', fname)
        keywords = data.get('keywords', [])
        lines    = data.get('lines', [])

        # 텍스트 수집
        kw_texts   = [kw if isinstance(kw, str) else kw.get('text','') for kw in keywords]
        line_texts = [ln.get('text','') for ln in lines]
        all_texts  = kw_texts + line_texts
        if not all_texts:
            print(f"[SKIP] {fname[:60]}"); continue

        embs = embed(all_texts, tokenizer, model_enc, device, prefix='query')

        # keywords → dict로 변환 + 라벨 주입
        new_keywords = []
        for i, kw in enumerate(keywords):
            text = kw if isinstance(kw, str) else kw.get('text', '')
            t1, s1, t2, s2 = top2(embs[i], label_kw_embs, active_labels)
            entry = {'text': text, 'top_label': t1, 'top_score': s1,
                     'second_label': t2, 'second_score': s2}
            if isinstance(kw, dict):
                entry = {**kw, **entry}
            new_keywords.append(entry)

        # lines → 라벨 필드 추가
        offset = len(kw_texts)
        new_lines = []
        for i, ln in enumerate(lines):
            t1, s1, t2, s2 = top2(embs[offset + i], label_kw_embs, active_labels)
            new_lines.append({**ln, 'top_label': t1, 'top_score': s1,
                               'second_label': t2, 'second_score': s2})

        data['keywords'] = new_keywords
        data['lines']    = new_lines

        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[{album[:45]}]  kw={len(new_keywords)}  lines={len(new_lines)}")

    print(f"\n{'='*60}\n✓ 완료")


if __name__ == '__main__':
    main()
