import json, os, re, sys, torch
import numpy as np
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_training_script"))
import config
from transformers import AutoTokenizer, AutoModel

SRC_DIR  = 'intent_release/02_intent_extraction/'
BASE_OUT = 'gap_output_puremodel/'
OUT_LINE = BASE_OUT + 'doc_probs_line/'
OUT_FULL = BASE_OUT + 'doc_probs_fulltext/'

DESC_FILE  = os.path.join(os.path.dirname(__file__), 'label_descriptions.json')
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


def score(text_emb, label_kw_embs, active_labels):
    """라벨 스코어 = 키워드별 raw cosine 최대값 → [0.70, 0.92] → [0, 1] 정규화"""
    result = {}
    for l, kw_embs in zip(active_labels, label_kw_embs):
        sims = np.dot(kw_embs, text_emb)
        result[l] = normalize(sims.max())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--desc', default=DESC_FILE)
    args = p.parse_args()

    with open(args.desc, encoding='utf-8') as f:
        label_desc_map = json.load(f)

    all_labels    = config.LABELS
    active_labels = [l for l in all_labels if label_desc_map.get(l, '').strip()]
    excluded_lbl  = [l for l in all_labels if l not in active_labels]
    print(f"활성 라벨 ({len(active_labels)}개): {active_labels}")
    print(f"제외 라벨 ({len(excluded_lbl)}개): {excluded_lbl}")

    for d in [OUT_LINE, OUT_FULL]:
        os.makedirs(d, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_enc = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model_enc.eval()
    print(f"\n순수 ME5 로드 완료 ({device}) — raw cosine, 노이즈 처리 없음\n{'='*65}")

    label_kw_embs = []
    for l in active_labels:
        kws  = [k.strip() for k in label_desc_map[l].split(',') if k.strip()]
        embs = embed(kws, tokenizer, model_enc, device, prefix='passage')
        label_kw_embs.append(embs)
        print(f"  {l} ({len(kws)}개): {kws}")
    print()

    meta_base = {
        'method':          'pure_ME5_norm_cosine_max_kw',
        'norm_min':        NORM_MIN,
        'norm_max':        NORM_MAX,
        'active_labels':   active_labels,
        'excluded_labels': excluded_lbl,
    }

    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(SRC_DIR, fname), encoding='utf-8') as f:
            data = json.load(f)

        album    = data.get('album', fname)
        keywords = data.get('keywords', [])
        lines    = data.get('lines', [])

        items = [{'type': 'keyword', 'text': kw.strip()} for kw in keywords if kw.strip()]
        items += [{'type': 'line', 'page': ln.get('page'), 'text': ln['text'].strip()}
                  for ln in lines if ln.get('text', '').strip()]

        if not items:
            print(f"[SKIP] {fname[:50]}\n"); continue

        texts     = [it['text'] for it in items]
        text_embs = embed(texts, tokenizer, model_enc, device, prefix='query')

        # 항목별 스코어
        item_results, all_arrs = [], []
        for i, it in enumerate(items):
            s = score(text_embs[i], label_kw_embs, active_labels)
            top_label = max(s, key=s.get)
            item_results.append({**it, 'top_label': top_label, 'scores': s})
            all_arrs.append(np.array([s[l] for l in active_labels]))

        # 전체 평균
        mean_arr  = np.mean(all_arrs, axis=0)
        scores_avg = {l: round(float(v), 4) for l, v in zip(active_labels, mean_arr)}

        # fulltext 한방
        full_emb    = embed([' '.join(texts)], tokenizer, model_enc, device, prefix='query')[0]
        scores_full = score(full_emb, label_kw_embs, active_labels)

        safe = re.sub(r'_+', '_',
                      re.sub(r'[^\w가-힣]', '_', fname.replace('.json', ''))).strip('_')[:60]
        meta = {**meta_base, 'source_file': fname, 'album': album}

        with open(os.path.join(OUT_LINE, f'{safe}.json'), 'w', encoding='utf-8') as f:
            json.dump({**meta, 'n_total': len(items),
                       'scores': scores_avg, 'per_item': item_results},
                      f, ensure_ascii=False, indent=2)

        with open(os.path.join(OUT_FULL, f'{safe}.json'), 'w', encoding='utf-8') as f:
            json.dump({**meta, 'n_chars': len(' '.join(texts)),
                       'scores': scores_full},
                      f, ensure_ascii=False, indent=2)

        def top4(d):
            return '  '.join(f'{l}={v:.4f}'
                             for l, v in sorted(d.items(), key=lambda x: -x[1])[:4])

        print(f"[{album[:40]}]  ({len(items)}개)")
        print(f"  line평균 : {top4(scores_avg)}")
        print(f"  fulltext : {top4(scores_full)}")
        print()

    print(f"{'='*65}\n✓ {BASE_OUT} 저장 완료")


if __name__ == '__main__':
    main()
