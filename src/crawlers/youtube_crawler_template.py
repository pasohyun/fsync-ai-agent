"""
유튜브 댓글 크롤러 — 엔시티위시 (귀여움 라벨)
YouTube Data API v3 사용
"""

import os
import json
import time
import argparse
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────
API_KEY = os.environ.get("YOUTUBE_API_KEY", "여기에_API_키_입력")

ARTIST   = "엔시티위시"
LABEL    = "cute"
PURPOSE  = "train"  # train | gap_target

# 엔시티위시 영상 목록 (video_id, video_type)
# → 아래 VIDEO_LIST를 직접 채워서 쓰세요
VIDEO_LIST = [
    # MV
    {"video_id": "영상ID_1", "video_type": "MV",      "title": "Wishful (MV)"},
    {"video_id": "영상ID_2", "video_type": "MV",      "title": "Steady (MV)"},
    # 티저
    {"video_id": "영상ID_3", "video_type": "teaser",  "title": "Wishful Teaser"},
    # 직캠
    {"video_id": "영상ID_4", "video_type": "fancam",  "title": "소희 직캠"},
    {"video_id": "영상ID_5", "video_type": "fancam",  "title": "유우시 직캠"},
    # 비하인드
    {"video_id": "영상ID_6", "video_type": "behind",  "title": "비하인드 영상"},
]

TARGET_COUNT  = 25_000   # 목표 댓글 수 (2~3만)
OUTPUT_DIR    = "data/train/cute"
OUTPUT_FILE   = f"{OUTPUT_DIR}/nct_wish_comments.jsonl"

# ────────────────────────────────────────────
# 크롤러
# ────────────────────────────────────────────

def get_video_info(youtube, video_id: str) -> dict:
    """영상 업로드 날짜 가져오기"""
    try:
        res = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        if res["items"]:
            return {
                "published_at": res["items"][0]["snippet"]["publishedAt"],
                "title": res["items"][0]["snippet"]["title"]
            }
    except HttpError as e:
        print(f"  [영상 정보 오류] {e}")
    return {"published_at": None, "title": None}


def fetch_comments(youtube, video: dict, max_count: int) -> list:
    """
    단일 영상의 댓글을 수집
    max_count: 이 영상에서 최대 몇 개 수집할지
    """
    video_id   = video["video_id"]
    video_type = video["video_type"]
    title      = video.get("title", "")

    print(f"\n▶ [{video_type}] {title} ({video_id})")

    # 영상 업로드 날짜 조회
    info = get_video_info(youtube, video_id)
    video_published_at = info["published_at"]

    comments   = []
    page_token = None
    fetched    = 0

    while fetched < max_count:
        try:
            params = dict(
                part        = "snippet",
                videoId     = video_id,
                maxResults  = 100,          # API 최대값
                textFormat  = "plainText",
                order       = "relevance",  # relevance | time
            )
            if page_token:
                params["pageToken"] = page_token

            res = youtube.commentThreads().list(**params).execute()

        except HttpError as e:
            if e.resp.status == 403:
                print(f"  [댓글 비활성화 또는 권한 없음] 건너뜀")
            elif e.resp.status == 429:
                print(f"  [할당량 초과] 60초 대기 후 재시도")
                time.sleep(60)
                continue
            else:
                print(f"  [API 오류] {e}")
            break

        for item in res.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id"          : video_id,
                "video_type"        : video_type,
                "video_title"       : title,
                "video_published_at": video_published_at,
                "artist"            : ARTIST,
                "label"             : LABEL,
                "purpose"           : PURPOSE,
                "comment_id"        : item["snippet"]["topLevelComment"]["id"],
                "text"              : top["textDisplay"].strip(),
                "likes"             : top["likeCount"],
                "published_at"      : top["publishedAt"],   # ISO 8601
                "crawled_at"        : datetime.now(timezone.utc).isoformat(),
            })
            fetched += 1
            if fetched >= max_count:
                break

        page_token = res.get("nextPageToken")
        if not page_token:
            break

        time.sleep(0.5)   # API 레이트 리밋 방지

    print(f"  → {fetched:,}개 수집 완료")
    return comments


def run(target_total: int = TARGET_COUNT):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    youtube = build("youtube", "v3", developerKey=API_KEY)

    total_collected = 0
    per_video_target = max(500, target_total // len(VIDEO_LIST))

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for video in VIDEO_LIST:
            if total_collected >= target_total:
                print(f"\n목표 {target_total:,}개 달성. 종료.")
                break

            remaining = target_total - total_collected
            limit     = min(per_video_target, remaining)

            comments = fetch_comments(youtube, video, max_count=limit)

            for c in comments:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

            total_collected += len(comments)
            print(f"  누적: {total_collected:,} / {target_total:,}")

            time.sleep(1)

    print(f"\n✅ 최종 수집: {total_collected:,}개 → {OUTPUT_FILE}")


# ────────────────────────────────────────────
# 유틸: 수집 현황 확인
# ────────────────────────────────────────────

def stats():
    """수집된 JSONL 파일 현황 출력"""
    if not os.path.exists(OUTPUT_FILE):
        print("아직 수집된 파일 없음")
        return

    counts = {}
    total  = 0
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            key = f"{d['video_type']} | {d['video_title']}"
            counts[key] = counts.get(key, 0) + 1
            total += 1

    print(f"\n총 {total:,}개\n")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {v:>6,}개  {k}")


# ────────────────────────────────────────────
# 유틸: 중복 제거
# ────────────────────────────────────────────

def dedup():
    """comment_id 기준 중복 제거"""
    if not os.path.exists(OUTPUT_FILE):
        print("파일 없음")
        return

    seen  = set()
    lines = []
    dup   = 0
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cid = d["comment_id"]
            if cid in seen:
                dup += 1
            else:
                seen.add(cid)
                lines.append(line)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"중복 {dup}개 제거 → {len(lines):,}개 남음")


# ────────────────────────────────────────────
# 실행
# ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   default="run",   choices=["run", "stats", "dedup"])
    parser.add_argument("--target", default=25000,   type=int, help="목표 댓글 수")
    args = parser.parse_args()

    if args.mode == "run":
        run(args.target)
    elif args.mode == "stats":
        stats()
    elif args.mode == "dedup":
        dedup()
