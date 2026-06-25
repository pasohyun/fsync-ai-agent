"""keywords를 dict→string으로, lines에서 라벨 필드 제거"""
import json, os

SRC_DIR = 'intent_release/02_intent_extraction/'

for fname in os.listdir(SRC_DIR):
    if not fname.endswith('.json'):
        continue
    fpath = os.path.join(SRC_DIR, fname)
    with open(fpath, encoding='utf-8') as f:
        data = json.load(f)

    # keywords: dict → string
    new_kw = []
    for kw in data.get('keywords', []):
        if isinstance(kw, dict):
            new_kw.append(kw.get('text', ''))
        else:
            new_kw.append(kw)
    data['keywords'] = new_kw

    # lines: 라벨 필드 제거
    new_lines = []
    for ln in data.get('lines', []):
        new_lines.append({k: v for k, v in ln.items()
                          if k not in ('top_label','top_score','second_label','second_score')})
    data['lines'] = new_lines

    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ {fname[:60]}")

print("\n원복 완료")
