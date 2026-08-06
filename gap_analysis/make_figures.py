"""발명내용 설명서 도면 재생성.

  --fig 1      도면 1 — 전체 파이프라인 구성도(S110~S170) 및 조건부 피드백 경로
  --fig 2      도면 2 — 실시예: 라벨별 상대 강도(임계값 0.80, 최종 9라벨)
  --fig trend  라벨별 일별 반응 추이 그래프 9장

Usage:
  python gap_analysis/make_figures.py --fig 1 --fig 2
  python gap_analysis/make_figures.py --fig trend --outdir 05_최종결과/도면
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 한글 라벨이 깨지지 않도록 설치된 CJK 폰트 중 하나를 선택
from matplotlib import font_manager

_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
for _cand in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    if _cand in _INSTALLED:
        matplotlib.rcParams["font.family"] = _cand
        break
else:
    print("[경고] 한글 폰트를 찾지 못했습니다. 라벨이 깨질 수 있습니다.")
matplotlib.rcParams["axes.unicode_minus"] = False

DEFAULT_LIFT  = "04_reaction/환절기/lift/hwanjeolgi_gap_lift_080_9label.json"
DEFAULT_TREND = "05_최종결과/반응추이(일별 그래프 9개)(히트수 라벨로 그래프 그리면됨)/hwanjeolgi_reaction_080_9label_by_date.json"
DEFAULT_OUT   = "05_최종결과/도면"

C_BOX    = "#f4f6fa"
C_EDGE   = "#33475b"
C_ACCENT = "#1f4e79"
C_UP     = "#c0392b"   # 통상 수준 초과
C_DOWN   = "#7f8c8d"   # 통상 수준 미달
C_INTENT = "#1f4e79"   # 의도 라벨


# ── 도면 1 ────────────────────────────────────────────────────────
def _box(ax, x, y, w, h, step, title, body, accent=False):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.6, edgecolor=C_ACCENT if accent else C_EDGE,
        facecolor="#e8eef7" if accent else C_BOX, zorder=2))
    ax.text(x, y + h/2 - 0.24, step, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_ACCENT, zorder=3)
    ax.text(x, y + 0.02, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1a1a1a", zorder=3)
    ax.text(x, y - h/2 + 0.24, body, ha="center", va="center",
            fontsize=8.2, color="#4a5568", zorder=3)


def _arrow(ax, p1, p2, dashed=False, rad=0.0, label=None):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color="#8b0000" if dashed else C_EDGE,
        linestyle=(0, (5, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, fontsize=8, color="#8b0000",
                ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


def fig1(out_path, dpi):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 14.2); ax.set_ylim(1.2, 13.0); ax.axis("off")

    W, H = 3.6, 1.05
    CX = 5.0          # 본류 컬럼 중심
    RAIL = 1.5        # 피드백 경로 세로 레일 x좌표

    _box(ax, CX, 12.0, W, H, "S110", "반응 데이터 수집·정제",
         "YouTube Data API / 언어·중복·스팸 필터")
    _box(ax, CX, 10.3, W, H, "S120", "반응 라벨 체계 유도",
         "임베딩·클러스터링 + LLM 명명 → K개 라벨", accent=True)
    _box(ax, CX, 8.6, W, H, "S130", "이중 라벨링·분류기 학습",
         "LLM 약지도 + 인간 골드셋 / 시그모이드 BCE")
    _box(ax, 3.5, 6.6, 3.2, H, "S140", "기준선 반응 분포",
         "학습 코퍼스 전체 · τ=0.80")
    _box(ax, 7.1, 6.6, 3.2, H, "S150", "대상 반응 분포",
         "분석 대상 콘텐츠 · 동일 τ")
    _box(ax, 11.4, 8.6, 3.4, H, "S160", "의도 라벨 추출",
         "기획 문서 → 의도 라벨 + 강조 강도")
    _box(ax, CX, 4.4, 6.0, 1.15, "S170", "괴리도(Gap) 산출 및 리포팅",
         "lift(l) = p_target(l) / p_baseline(l)  →  전달 / 누수 / 자생 판정", accent=True)
    _box(ax, CX, 2.2, 6.0, H, "출력", "대시보드 · 보고서",
         "일별 반응 추이 · 동종 코호트 포지셔닝")

    # 기획 문서 입력
    ax.text(11.4, 10.5, "제작 주체 기획 문서\n(기획안·발매보고서·프로모션 계획)",
            ha="center", va="center", fontsize=8.6, color="#4a5568",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#b0b8c4", ls="--"))

    _arrow(ax, (CX, 11.42), (CX, 10.88))
    _arrow(ax, (CX, 9.72),  (CX, 9.18))
    _arrow(ax, (CX, 8.02),  (3.5, 7.18))
    _arrow(ax, (CX, 8.02),  (7.1, 7.18))
    _arrow(ax, (3.5, 6.02), (4.2, 5.03))
    _arrow(ax, (7.1, 6.02), (5.9, 5.03))
    _arrow(ax, (11.4, 10.1), (11.4, 9.18))
    _arrow(ax, (11.4, 8.02), (8.2, 4.7))
    _arrow(ax, (CX, 3.82),  (CX, 2.78))

    # 조건부 피드백 경로: 파라미터 변경 시 S120부터 재수행.
    # 박스를 가로지르지 않도록 좌측 레일로 우회시킨다.
    fb = dict(color="#8b0000", linestyle=(0, (5, 3)), linewidth=1.4, zorder=1)
    ax.plot([CX - 3.0, RAIL], [4.4, 4.4], **fb)           # S170 좌측 → 레일
    ax.plot([RAIL, RAIL], [4.4, 10.3], **fb)              # 레일 상승
    ax.add_patch(FancyArrowPatch(
        (RAIL, 10.3), (CX - W/2, 10.3), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color="#8b0000", linestyle=(0, (5, 3)), zorder=1))
    ax.text(RAIL - 0.55, 7.4,
            "조건부 피드백\n(라벨 체계·임계값 등 파라미터 변경 시 S120부터 자동 재수행)",
            fontsize=8.4, color="#8b0000", ha="center", va="center", rotation=90)

    ax.set_title("[도면 1] 전체 파이프라인 구성도(S110~S170) 및 조건부 피드백 경로",
                 fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ {out_path}")


# ── 도면 2 ────────────────────────────────────────────────────────
def fig2(lift_path, out_path, dpi):
    d = json.loads(Path(lift_path).read_text(encoding="utf-8"))
    items = sorted(d["labels"].items(), key=lambda x: x[1]["lift"])
    names = [k.replace("_", "·") for k, _ in items]
    lifts = [v["lift"] for _, v in items]
    intents = [v["intent"] for _, v in items]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = [C_INTENT if it != "none" else (C_UP if lf > 1 else C_DOWN)
              for lf, it in zip(lifts, intents)]
    bars = ax.barh(names, lifts, color=colors, height=0.62, zorder=3)

    ax.axvline(1.0, color="#333333", linewidth=1.4, zorder=4)
    ax.text(1.0, len(names) - 0.35, " 통상 수준 (lift = 1.0)",
            fontsize=8.6, color="#333333", va="center")

    for bar, lf, it in zip(bars, lifts, intents):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f"{lf:.2f}배", va="center", fontsize=9,
                fontweight="bold" if it != "none" else "normal",
                color=C_INTENT if it != "none" else "#4a5568", zorder=4)

    for tick, it in zip(ax.get_yticklabels(), intents):
        if it != "none":
            tick.set_color(C_INTENT)
            tick.set_fontweight("bold")

    ax.set_xlim(0, max(lifts) * 1.22)
    ax.set_xlabel("상대 강도  lift(l) = p_target(l) / p_baseline(l)", fontsize=10)
    ax.grid(axis="x", linestyle=":", color="#c8ccd4", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=C_INTENT),
               plt.Rectangle((0, 0), 1, 1, color=C_UP),
               plt.Rectangle((0, 0), 1, 1, color=C_DOWN)]
    ax.legend(handles, ["의도 라벨", "비의도 · 통상 초과", "비의도 · 통상 미달"],
              fontsize=8.6, loc="lower right", frameon=False)

    method = d.get("method", "")
    ax.set_title(f"[도면 2] 실시예 — 엔플라잉 '{d.get('release','')}' 릴리즈의 라벨별 상대 강도\n"
                 f"({method}, 의도 라벨 {d['n_success']}/{d['n_intended']} 전달)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ {out_path}")


# ── 일별 반응 추이 ─────────────────────────────────────────────────
def fig_trend(trend_path, out_dir, dpi):
    d = json.loads(Path(trend_path).read_text(encoding="utf-8"))
    by_date = d["by_date"]
    dates = sorted(by_date.keys())
    for label in d["labels"]:
        series = [by_date[dt]["hit_counts"].get(label, 0) for dt in dates]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(dates, series, marker="o", markersize=3.5,
                linewidth=1.7, color=C_ACCENT)
        ax.fill_between(range(len(dates)), series, alpha=0.12, color=C_ACCENT)
        ax.set_title(f"{label.replace('_', '·')} — 일별 반응 추이 (히트 수)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("히트 수", fontsize=9)
        ax.grid(axis="y", linestyle=":", color="#c8ccd4")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        step = max(1, len(dates) // 12)
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)],
                           rotation=45, ha="right", fontsize=8)
        fig.tight_layout()
        out = out_dir / f"trend_{label.replace('_', '-')}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✓ {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", action="append", default=None,
                   choices=["1", "2", "trend"], help="반복 지정 가능 (기본: 1,2)")
    p.add_argument("--lift",   default=DEFAULT_LIFT)
    p.add_argument("--trend",  default=DEFAULT_TREND)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    figs = args.fig or ["1", "2"]
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "1" in figs:
        fig1(out_dir / "도면1_파이프라인_구성도.png", args.dpi)
    if "2" in figs:
        fig2(args.lift, out_dir / "도면2_라벨별_상대강도.png", args.dpi)
    if "trend" in figs:
        fig_trend(args.trend, out_dir, args.dpi)


if __name__ == "__main__":
    main()
