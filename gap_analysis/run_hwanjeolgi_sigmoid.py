"""대상 콘텐츠(환절기) 반응 분포 산출 — 발명내용 설명서 5.(5)단계.

기준선(run_corpus_sigmoid.py)과 동일한 분류기·동일한 임계값·동일한 정규화 방식으로
대상 릴리즈 댓글의 라벨별 반응 분포를 산출한다. 두 분포가 동일 조건에서 나와야
compute_gap_lift.py 의 상대강도(lift) 대비가 성립한다.

Usage:
  python gap_analysis/run_hwanjeolgi_sigmoid.py
  python gap_analysis/run_hwanjeolgi_sigmoid.py --gpu 1 --batch 64
"""
import json, re, sys, os, argparse, torch
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

DEFAULT_SRC = "04_reaction/환절기/comments_nflying_hwanjeolgi_test_before_excl_killingvoice_2026-06-26.jsonl"
DEFAULT_OUT = "04_reaction/환절기/output"

# 분석 대상 라벨 집합. 9label = 최종본(노이즈·비주얼·역주행 제외)
LABELS_9 = ['장기_팬덤','보컬_라이브','신규_유입','밴드_정체성','위로_공감','연주_악기','이별_감성','청량_여름','음악성']
LABELS_8 = ['비주얼_멤버매력','보컬_라이브','밴드_정체성','위로_공감','연주_악기','이별_감성','청량_여름','음악성']
LABELS_7 = ['보컬_라이브','밴드_정체성','위로_공감','연주_악기','이별_감성','청량_여름','음악성']


def save_thresh(rows, label_set, out_path, thresh):
    hits, n_hit, n_multi = [], 0, 0
    for r in rows:
        matched = [l for l in label_set if r['sigmoid'].get(l, 0) > thresh]
        if matched:
            hits.extend(matched); n_hit += 1
            if len(matched) >= 2: n_multi += 1
    total  = len(hits); counts = Counter(hits)
    scores = {l: round(counts.get(l,0)/total, 4) if total else 0.0 for l in label_set}
    pcts   = {l: round(counts.get(l,0)/total*100, 2) if total else 0.0 for l in label_set}
    result = {'method': f'sigmoid_threshold_{thresh}', 'threshold': thresh,
              'n_total': len(rows), 'n_comments_hit': n_hit, 'n_multi_hit': n_multi,
              'n_label_hits': total, 'labels': label_set,
              'reaction_pct': scores, 'reaction_pct_percent': pcts,
              'hit_counts': {l: counts.get(l,0) for l in label_set}}
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    top3 = sorted(pcts.items(), key=lambda x: -x[1])[:3]
    print(f"  ✓ {out_path.name}  히트:{n_hit:,}  멀티:{n_multi}  top3:{top3}", flush=True)


def save_top1(rows, label_set, out_path):
    top1s  = [max(label_set, key=lambda l: r['sigmoid'].get(l, 0)) for r in rows]
    n      = len(top1s); counts = Counter(top1s)
    scores = {l: round(counts.get(l,0)/n, 4) if n else 0.0 for l in label_set}
    pcts   = {l: round(counts.get(l,0)/n*100, 2) if n else 0.0 for l in label_set}
    result = {'method': 'sigmoid_top1', 'n_total': len(rows), 'labels': label_set,
              'reaction_pct': scores, 'reaction_pct_percent': pcts,
              'hit_counts': {l: counts.get(l,0) for l in label_set}}
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    top3 = sorted(pcts.items(), key=lambda x: -x[1])[:3]
    print(f"  ✓ {out_path.name}  top3:{top3}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=DEFAULT_SRC)
    p.add_argument('--outdir', default=DEFAULT_OUT)
    p.add_argument('--prefix', default='hwanjeolgi', help='출력 파일명 접두어(릴리즈 슬러그)')
    p.add_argument('--ckpt', default='model_output/me5_large_v2/best_model.pt')
    p.add_argument('--model-name', default='intfloat/multilingual-e5-large')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch', type=int, default=128)
    args = p.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    out_sig = out_dir / f"{args.prefix}_me5_sigmoid.jsonl"

    # ── 모델 로드 ────────────────────────────────────────────────
    device    = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = RoBERTaMultiLabel(args.model_name, config.NUM_LABELS).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"모델 로드 완료 ({device})", flush=True)

    # ── 데이터 로드 + 한국어 필터 ────────────────────────────────
    raw = []
    with open(args.src, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    filtered = [r for r in raw if is_korean(r.get('text', ''))]
    print(f"전체 {len(raw):,} → 한국어 {len(filtered):,}개\n", flush=True)

    # ── sigmoid 추론 ─────────────────────────────────────────────
    B = args.batch
    results = []
    with open(out_sig, 'w', encoding='utf-8') as fout:
        for i in range(0, len(filtered), B):
            batch = [r['text'] for r in filtered[i:i+B]]
            enc   = tokenizer(batch, max_length=config.MAX_LEN, padding=True,
                              truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                logits = model(enc['input_ids'], enc['attention_mask'])
                sigs   = torch.sigmoid(logits).cpu().tolist()
            for row, sig_list in zip(filtered[i:i+B], sigs):
                rec = {
                    'comment_id':   row.get('comment_id'),
                    'video_title':  row.get('video_title'),
                    'text':         row['text'],
                    'sigmoid':      {l: round(v, 4) for l, v in zip(config.LABELS, sig_list)},
                    'published_at': row.get('published_at'),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                results.append(rec)
            if i % (B * 10) == 0:
                print(f"  {min(i+B, len(filtered)):,} / {len(filtered):,}", flush=True)

    print(f"\n✓ {out_sig} ({len(results):,}개)\n", flush=True)

    # ── 반응 분포 파일 ───────────────────────────────────────────
    for label_set, tag in [(LABELS_9,'9label'), (LABELS_8,'8label'), (LABELS_7,'7label')]:
        save_thresh(results, label_set, out_dir/f"{args.prefix}_reaction_080_{tag}.json", 0.80)
        save_thresh(results, label_set, out_dir/f"{args.prefix}_reaction_085_{tag}.json", 0.85)
        save_top1  (results, label_set, out_dir/f"{args.prefix}_reaction_top1_{tag}.json")

    print("\n✓ 대상 반응 분포 산출 완료", flush=True)


if __name__ == '__main__':
    main()
