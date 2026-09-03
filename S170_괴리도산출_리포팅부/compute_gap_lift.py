"""의도-반응 괴리도(Gap) 산출 — 발명내용 설명서 5.(7)단계 / 청구항 1·2·3.

기준선 반응 분포와 대상 반응 분포를 라벨별로 대비하여 상대 강도(lift)를 산출하고,
기획 문서에서 추출한 의도 라벨을 기준으로 괴리도를 판정한다.

    ④ lift(l) = p_target(l) / p_baseline(l)

lift > 1 이면 해당 반응이 통상 수준보다 강하게 발생, 1 미만이면 통상 수준 미달.
두 분포는 동일 라벨 체계·동일 분류기·동일 임계값·동일 정규화 방식으로 산출된
reaction json (S140/run_corpus_sigmoid.py, S150/run_hwanjeolgi_sigmoid.py 출력)이어야 한다.

Usage:
  # 최종본(임계값 0.80, 9라벨) 재현
  python S170_괴리도산출_리포팅부/compute_gap_lift.py

  # 기존 산출물과 일치하는지 검증
  python S170_괴리도산출_리포팅부/compute_gap_lift.py \
      --verify 04_reaction/환절기/lift/hwanjeolgi_gap_lift_080_9label.json

  # 8라벨 변형
  python S170_괴리도산출_리포팅부/compute_gap_lift.py \
      --baseline 04_reaction/전체/OUTPUT/corpus_reaction_08_8label.json \
      --target   04_reaction/환절기/output/hwanjeolgi_reaction_080_8label.json \
      --out      04_reaction/환절기/lift/hwanjeolgi_gap_lift_080_8label.json
"""
import argparse
import json
import math
from pathlib import Path

DEFAULT_BASELINE = "04_reaction/전체/OUTPUT/corpus_reaction_08_9label.json"
DEFAULT_TARGET   = "04_reaction/환절기/output/hwanjeolgi_reaction_080_9label.json"
DEFAULT_INTENT   = "02_intent/intent_labels/hwanjeolgi_intent.json"
DEFAULT_OUT      = "04_reaction/환절기/lift/hwanjeolgi_gap_lift_080_9label.json"

# 판정 경계.
#   - 의도 라벨은 ALIGN_CUT(=1.0) 기준으로 전달/누수를 가른다 (청구항 3).
#   - 비의도 라벨은 HIGH_CUT 이상이면 '자생' 반응으로 별도 표시한다.
#   - LOW_CUT(1.0)은 비의도 라벨의 '보통/낮음' 경계다. 통상 수준(lift=1.0)을
#     기준으로 삼아 ALIGN_CUT 과 동일하게 둔다. 필요 시 --low-cut 으로 조정.
ALIGN_CUT = 1.0
HIGH_CUT  = 1.5
LOW_CUT   = 1.0

V_ALIGN = "✅ 정렬"
V_LEAK  = "🔴 누수"
V_WILD  = "🟡 자생"
V_MID   = "  보통"
V_LOW   = "  낮음"


def verdict(lift, intent, align_cut=ALIGN_CUT, high_cut=HIGH_CUT, low_cut=LOW_CUT):
    """상대 강도와 의도 여부로 라벨 단위 괴리도를 판정한다 (청구항 3)."""
    if intent != "none":
        # 의도한 메시지가 실제로 전달되었는가
        return V_ALIGN if lift >= align_cut else V_LEAK
    # 의도하지 않았으나 강하게 발생했는가
    if lift >= high_cut:
        return V_WILD
    if lift >= low_cut:
        return V_MID
    return V_LOW


def load_dist(path):
    """reaction json에서 라벨별 반응 분포(reaction_pct)를 읽는다."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "reaction_pct" not in d:
        raise SystemExit(f"[에러] {path} 에 reaction_pct 가 없습니다.")
    return d["reaction_pct"], d


def load_intent(path):
    """의도 라벨 파일에서 {라벨: 강조강도} 를 읽는다 (설명서 5.(6) 산출물)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "intended_labels" in d:
        return d["intended_labels"]
    # step1_build_intent_vectors.py 형식 호환: level 점수가 0보다 큰 라벨만
    if "intent_scores" in d:
        return {l: "high" for l, s in d["intent_scores"].items() if s > 0}
    raise SystemExit(f"[에러] {path} 에 intended_labels 가 없습니다.")


def compute(baseline_pct, target_pct, intent_map, release, release_key,
            method, align_cut, high_cut, low_cut):
    labels = {}
    for label, p_target in target_pct.items():
        p_base = baseline_pct.get(label)
        if p_base is None:
            raise SystemExit(f"[에러] 기준선 분포에 '{label}' 라벨이 없습니다. "
                             f"동일 라벨 체계의 분포끼리만 대비할 수 있습니다.")
        intent = intent_map.get(label, "none")
        if p_base == 0:
            lift = float("inf") if p_target > 0 else 0.0
            log2 = None
        else:
            lift = p_target / p_base
            log2 = round(math.log2(lift), 3) if lift > 0 else None
        labels[label] = {
            release_key: p_target,
            "baseline":  p_base,
            "lift":      round(lift, 3),
            "log2_lift": log2,
            "intent":    intent,
            "verdict":   verdict(lift, intent, align_cut, high_cut, low_cut),
        }
        # 명세서 예시 출력 순서(baseline → target)에 맞춰 키 정렬
        labels[label] = {
            "baseline":  labels[label]["baseline"],
            release_key: labels[label][release_key],
            "lift":      labels[label]["lift"],
            "log2_lift": labels[label]["log2_lift"],
            "intent":    labels[label]["intent"],
            "verdict":   labels[label]["verdict"],
        }

    # 의도 라벨 중 전달에 성공한 비율로 스칼라 괴리도를 산출
    intended = [l for l in labels if labels[l]["intent"] != "none"]
    n_intended = len(intended)
    n_success  = sum(1 for l in intended if labels[l]["lift"] >= align_cut)
    gap_scalar = round(1 - n_success / n_intended, 4) if n_intended else None

    return {
        "release":    release,
        "method":     method,
        "intent":     {l: labels[l]["intent"] for l in intended},
        "gap_scalar": gap_scalar,
        "n_success":  n_success,
        "n_intended": n_intended,
        "labels":     labels,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=DEFAULT_BASELINE, help="기준선 반응 분포 json")
    p.add_argument("--target",   default=DEFAULT_TARGET,   help="대상 반응 분포 json")
    p.add_argument("--intent",   default=DEFAULT_INTENT,   help="의도 라벨 json")
    p.add_argument("--out",      default=DEFAULT_OUT)
    p.add_argument("--release",     default="환절기")
    p.add_argument("--release-key", default="hwanjeolgi", help="labels 내 대상 분포 키 이름")
    p.add_argument("--method", default=None, help="미지정 시 대상 분포에서 자동 생성")
    p.add_argument("--align-cut", type=float, default=ALIGN_CUT)
    p.add_argument("--high-cut",  type=float, default=HIGH_CUT)
    p.add_argument("--low-cut",   type=float, default=LOW_CUT)
    p.add_argument("--verify", default=None, help="기존 산출물과 비교만 하고 저장하지 않음")
    args = p.parse_args()

    baseline_pct, _        = load_dist(args.baseline)
    target_pct, target_doc = load_dist(args.target)
    intent_map             = load_intent(args.intent)

    method = args.method
    if method is None:
        thresh = target_doc.get("threshold")
        n_lab  = len(target_doc.get("labels", target_pct))
        method = f"sigmoid_{thresh:.2f}_{n_lab}label" if thresh else f"{n_lab}label"

    result = compute(baseline_pct, target_pct, intent_map, args.release,
                     args.release_key, method,
                     args.align_cut, args.high_cut, args.low_cut)

    for label, v in sorted(result["labels"].items(), key=lambda x: -x[1]["lift"]):
        mark = "*" if v["intent"] != "none" else " "
        print(f"{mark} {label:<12s} lift={v['lift']:>6.3f}  "
              f"intent={v['intent']:<5s} {v['verdict']}")
    print(f"\ngap_scalar={result['gap_scalar']}  "
          f"({result['n_success']}/{result['n_intended']} 의도 라벨 전달)")

    if args.verify:
        expected = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        if expected == result:
            print(f"\n✓ 검증 통과: {args.verify} 와 완전히 일치")
        else:
            print(f"\n✗ 검증 실패: {args.verify} 와 불일치")
            for key in ("release", "method", "gap_scalar", "n_success", "n_intended", "intent"):
                if expected.get(key) != result.get(key):
                    print(f"  [{key}] 기존={expected.get(key)} / 신규={result.get(key)}")
            for label in set(expected.get("labels", {})) | set(result["labels"]):
                e, r = expected.get("labels", {}).get(label), result["labels"].get(label)
                if e != r:
                    print(f"  [{label}] 기존={e}\n              신규={r}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {out_path} 저장 완료")


if __name__ == "__main__":
    main()
