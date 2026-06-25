import json
import re

HANGUL_RE = re.compile(r"[가-힣]")

def has_enough_korean(text: str, min_ratio: float = 0.2) -> bool:
    if not text:
        return False
    return len(HANGUL_RE.findall(text)) / len(text) >= min_ratio

input_path = "../data/comments.jsonl"
output_path = "../data/comments_korean.jsonl"

total = 0
kept = 0

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        total += 1
        obj = json.loads(line)
        if has_enough_korean(obj.get("text", "")):
            fout.write(line + "\n")
            kept += 1

print(f"전체: {total:,}개")
print(f"한국어 댓글 (비율 ≥ 20%): {kept:,}개")
print(f"비율: {kept/total*100:.1f}%")
print(f"저장 위치: {output_path}")
