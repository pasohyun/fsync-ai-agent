import json
import os
import time
import anthropic

LABELS = ["섹시", "응원", "입덕", "퍼포먼스", "유머", "재방문", "기타"]

SYSTEM_PROMPT = """당신은 K-pop 팬 댓글 분류 전문가입니다.
주어진 댓글이 아래 라벨 중 하나 이상에 해당하면 표시하세요 (중복 가능).

라벨 정의:
- 섹시: 아이돌의 매력/섹시함/외모/신체에 직접 반응하는 댓글 (예: "양기가", "게이될것같다", "얼굴이 너무 섹시")
- 응원: 아이돌에 대한 일방적인 사랑·찬양·응원 표현 (예: "사랑해", "최고다", "항상 응원할게요")
- 입덕: 새로 팬이 됨을 선언하거나 입덕 과정 서술 (예: "오늘부터 1일", "나왜이제야몬엑알지", "결국 인정해버림")
- 퍼포먼스: 무대·안무·노래·비주얼에 대한 구체적 평가/감상 (예: "춤이 너무 잘어울려", "노래 춤 착장 다 좋음")
- 유머: 밈·드립·유머러스한 반응 (예: "구청에서보자", "귀신 쫓을 때 틀어두면", 과장된 반응)
- 재방문: 반복 시청, 중독성 언급, 자주 돌아온다는 댓글 (예: "또 보러왔어", "오늘도 여기")
- 기타: 질문, 일상 잡담, 위 분류에 해당하지 않는 댓글

출력 형식 (반드시 이 형식만):
각 댓글에 대해 한 줄씩: <인덱스>|<라벨1>,<라벨2>,...
예시:
0|섹시,응원
1|유머
2|기타"""

def label_batch(client, batch: list[dict]) -> dict[int, list[str]]:
    lines = "\n".join(f"{r['idx']}|{r['text']}" for r in batch)
    user_msg = f"다음 댓글들을 분류해주세요:\n\n{lines}"

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    results = {}
    for line in resp.content[0].text.strip().splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        idx_str, labels_str = line.split("|", 1)
        try:
            idx = int(idx_str.strip())
            labels = [l.strip() for l in labels_str.split(",") if l.strip() in LABELS]
            if not labels:
                labels = ["기타"]
            results[idx] = labels
        except ValueError:
            continue
    return results


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 환경변수를 설정해주세요.")

    client = anthropic.Anthropic(api_key=api_key)

    with open("to_label.jsonl") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    for i, r in enumerate(rows):
        r["idx"] = i

    batch_size = 50
    all_labels: dict[int, list[str]] = {}

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        print(f"배치 {start // batch_size + 1}/{(len(rows) + batch_size - 1) // batch_size} 처리 중... ({start}~{start+len(batch)-1})")
        try:
            result = label_batch(client, batch)
            all_labels.update(result)
        except Exception as e:
            print(f"  오류 발생: {e}, 재시도...")
            time.sleep(5)
            result = label_batch(client, batch)
            all_labels.update(result)
        time.sleep(0.3)

    with open("comments_labeled.jsonl", "w", encoding="utf-8") as fout:
        for r in rows:
            i = r["idx"]
            labels = all_labels.get(i, ["기타"])
            out = {k: v for k, v in r.items() if k != "idx"}
            out["labels"] = labels
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\n완료! comments_labeled.jsonl 저장")

    from collections import Counter
    counter = Counter()
    for labels in all_labels.values():
        for l in labels:
            counter[l] += 1
    print("\n라벨 분포:")
    for label, count in counter.most_common():
        print(f"  {label}: {count}개 ({count/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
