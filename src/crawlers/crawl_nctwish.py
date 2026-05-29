"""
엔시티위시 유튜브 댓글 수집기
영상 3개 대상으로 먼저 테스트 수집
"""

import os, json, time
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ── API 키 설정 ──────────────────────────────
API_KEY = os.environ.get("YOUTUBE_API_KEY", "")   # 환경변수로 넣거나 아래에 직접 입력
# API_KEY = "AIza..."                              # 직접 입력 시 이 줄 주석 해제

# ── 수집 대상 영상 ────────────────────────────
VIDEO_LIST = [
    {"video_id": "oorVGzxTzuw", "video_type": "MV", "title": "엔시티위시_영상1"},
    {"video_id": "pqM9nI0z48c", "video_type": "MV", "title": "엔시티위시_영상2"},
    {"video_id": "hvQZs3k6Ytk", "video_type": "MV", "title": "엔시티위시_영상3"},
]

# ── 설정 ──────────────────────────────────────
ARTIST          = "엔시티위시"
LABEL           = "cute"
PURPOSE         = "train"
TARGET_PER_VIDEO = 8_000    # 영상당 최대 수집 (3개 × 8000 = 24,000개)
OUTPUT_DIR      = "data/train/cute"
OUTPUT_FILE     = f"{OUTPUT_DIR}/nct_wish_comments.jsonl"

# ─────────────────────────────────────────────

def get_video_info(youtube, video_id):
    try:
        res = youtube.videos().list(part="snippet", id=video_id).execute()
        if res["items"]:
            snippet = res["items"][0]["snippet"]
            return {
                "published_at": snippet["publishedAt"],
                "title":        snippet["title"],
            }
    except HttpError:
        pass
    return {"published_at": None, "title": None}


def fetch_comments(youtube, video, max_count):
    video_id   = video["video_id"]
    video_type = video["video_type"]

    # 실제 영상 제목 가져오기
    info  = get_video_info(youtube, video_id)
    title = info["title"] or video.get("title", video_id)
    video_published_at = info["published_at"]

    print(f"\n▶ [{video_type}] {title}")
    print(f"  목표: {max_count:,}개")

    comments   = []
    page_token = None

    while len(comments) < max_count:
        try:
            params = dict(
                part       = "snippet",
                videoId    = video_id,
                maxResults = 100,
                textFormat = "plainText",
                order      = "relevance",
            )
            if page_token:
                params["pageToken"] = page_token

            res = youtube.commentThreads().list(**params).execute()

        except HttpError as e:
            if e.resp.status == 403:
                print(f"  ⚠️  댓글 비활성화 또는 접근 불가 — 건너뜀")
            elif e.resp.status == 429:
                print(f"  ⏳ 할당량 초과 — 60초 대기")
                time.sleep(60)
                continue
            else:
                print(f"  ❌ API 오류: {e}")
            break

        for item in res.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            text = top["textDisplay"].strip()
            if not text:
                continue

            comments.append({
                "video_id"          : video_id,
                "video_type"        : video_type,
                "video_title"       : title,
                "video_published_at": video_published_at,
                "artist"            : ARTIST,
                "label"             : LABEL,
                "purpose"           : PURPOSE,
                "comment_id"        : item["snippet"]["topLevelComment"]["id"],
                "text"              : text,
                "likes"             : top["likeCount"],
                "published_at"      : top["publishedAt"],
                "crawled_at"        : datetime.now(timezone.utc).isoformat(),
            })

            if len(comments) >= max_count:
                break

        page_token = res.get("nextPageToken")
        if not page_token:
            print(f"  ℹ️  더 이상 댓글 없음 (총 {len(comments):,}개)")
            break

        # 진행상황 출력 (500개마다)
        if len(comments) % 500 == 0:
            print(f"  수집 중... {len(comments):,}개")

        time.sleep(0.5)

    print(f"  ✅ 완료: {len(comments):,}개")
    return comments, title


def run():
    if not API_KEY:
        print("❌ API 키가 없습니다. API_KEY를 설정해주세요.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    youtube = build("youtube", "v3", developerKey=API_KEY)

    total = 0
    seen_ids = set()

    # 기존 파일에서 중복 방지용 ID 로드
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                seen_ids.add(d["comment_id"])
        print(f"기존 파일 로드: {len(seen_ids):,}개 comment_id 확인")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for video in VIDEO_LIST:
            comments, _ = fetch_comments(youtube, video, TARGET_PER_VIDEO)

            new = 0
            for c in comments:
                if c["comment_id"] not in seen_ids:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    seen_ids.add(c["comment_id"])
                    new += 1

            total += new
            print(f"  신규 저장: {new:,}개 | 누적: {total:,}개")
            time.sleep(1)

    print(f"\n{'='*40}")
    print(f"최종 수집 완료: {total:,}개")
    print(f"저장 위치: {OUTPUT_FILE}")


def stats():
    if not os.path.exists(OUTPUT_FILE):
        print("수집 파일 없음")
        return
    counts = {}
    total  = 0
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            k = d["video_title"]
            counts[k] = counts.get(k, 0) + 1
            total += 1
    print(f"\n총 {total:,}개")
    for k, v in counts.items():
        bar = "█" * (v // 100)
        print(f"  {v:>6,}개  {k}  {bar}")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "stats":
        stats()
    else:
        run()
