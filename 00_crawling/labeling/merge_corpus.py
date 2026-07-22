#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""통합 크롤링 코퍼스 빌더: 4명 팀원의 '한글 최종 필터링' 댓글을 하나로 합침.

팀원별 파일명 규칙이 제각각이라(ko/korean/filtered/final/band_all) 사용할 파일을
명시적 매니페스트로 고정한다. (자동탐색이 틀리면 더 위험 — 구조는 팀 4인/8아티스트로 고정)
- 한글 필터 안 된 raw(comments.jsonl)·비한글(nonko)·중복 합본 원본(Day6/LUCY)은 제외
- comment_id 기준 중복 제거(재크롤 중복 정리), 원본 필드·빈 label 칸 그대로 보존
- gold 라벨셋은 코퍼스의 부분집합이라 여기선 제외하지 않음(학습셋 구성 단계에서 분리)

  python3 src/labeling/merge_corpus.py
출력: data/train/band/band_korean_all.jsonl  (+ 분포 리포트 stdout)
"""
import json
import os
import re
import sys
from collections import Counter, OrderedDict

ROOT = "data/train/band"
OUT = os.path.join(ROOT, "band_korean_all.jsonl")

# 사용할 '한글 최종 필터링' 파일 매니페스트 (소스가 늘면 여기 추가)
INPUTS = [
    "jimin/comments_nflying_ko.jsonl",                 # 엔플라잉 (nonko 제외)
    "juhyeong/band_all.jsonl",                         # Day6+LUCY 합본 (개별 파일 제외)
    "sohyun/dragon_pony_comments_filtered.jsonl",      # 드래곤포니
    "sohyun/xdinary_heroes_comments_final.jsonl",      # 엑스디너리히어로즈
    "soobin/data_cnblue/comments_korean.jsonl",        # CNBLUE (raw 제외)
    "soobin/data_ftisland/comments_korean.jsonl",      # FTISLAND
    "soobin/data_qwer/comments_korean.jsonl",          # QWER
]

HANGUL = re.compile(r"[가-힣]")
# 보존할 표준 키 순서(없으면 기본값). 알 수 없는 키는 뒤에 그대로 붙임.
STD_KEYS = ["comment_id", "artist", "video_id", "video_type", "video_title",
            "video_published_at", "purpose", "text", "likes",
            "published_at", "crawled_at", "label"]


def norm_row(d):
    """키 순서 표준화 + label 칸 보장. 원본 값은 손대지 않음."""
    out = OrderedDict()
    for k in STD_KEYS:
        if k in d:
            out[k] = d[k]
    for k, v in d.items():          # 표준 외 키도 보존
        if k not in out:
            out[k] = v
    # label 정규화: 빈 문자열/None/리스트아님 → [] (소스별 "" vs [] 표현 통일)
    lab = out.get("label", [])
    out["label"] = lab if isinstance(lab, list) and lab else []
    return out


def main():
    seen = set()
    rows = []
    per_file = Counter()
    art = Counter()
    dup = 0
    no_id = 0
    no_ko = 0

    for rel in INPUTS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print("  ⚠ 누락(건너뜀):", path)
            continue
        cnt = 0
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            cid = str(d.get("comment_id", "")).strip()
            txt = str(d.get("text", "")).strip()
            if not HANGUL.search(txt):       # 안전장치: 한글 없는 행 제외
                no_ko += 1
                continue
            if not cid:
                no_id += 1
                key = "TXT::" + re.sub(r"\s+", " ", txt).lower()
            else:
                key = "ID::" + cid
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            rows.append(norm_row(d))
            art[d.get("artist", "?")] += 1
            cnt += 1
        per_file[rel] = cnt

    os.makedirs(ROOT, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 64)
    print(" 통합 코퍼스 생성:", OUT)
    print("=" * 64)
    print(" 입력 파일 %d개 → 고유 댓글 %d개" % (len(per_file), len(rows)))
    print(" 제거: 중복 comment_id %d, 한글없음 %d  (빈 id %d건은 text로 dedup)" % (dup, no_ko, no_id))
    print("\n 아티스트별 분포:")
    for a, n in art.most_common():
        print("   %-20s %6d  (%.1f%%)" % (a, n, 100 * n / len(rows)))
    print("\n 파일별 기여:")
    for f, n in per_file.items():
        print("   %-46s %6d" % (f, n))
    print("=" * 64)


if __name__ == "__main__":
    main()
