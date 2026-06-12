import json
import re
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build

API_KEY = "AIzaSyCwp6UDRNswiy7--lvYbllSJc615dbYh1Y"

# ── 고정값 ────────────────────────────────────────────────────
ARTIST = "FTISLAND"
OUTPUT = "../data/comments.jsonl"
MAX_COMMENTS = 999999
# ─────────────────────────────────────────────────────────────

# ── 몬스타 엑스 ─────────────────
#VIDEOS = [
    # {"url": "https://www.youtube.com/watch?v=-ToHbHcolfA&list=RD-ToHbHcolfA&start_radio=1", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=2zXg8CbymYc&list=RD2zXg8CbymYc&start_radio=1", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=4PGX1ZYvzvo&list=RD4PGX1ZYvzvo&start_radio=1", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=3C3hIJg4rHo&list=PLETayfJXr36BRH1_wjI3ypJwK6Zlzzjjg&index=14", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=aHKXYsQfHu8", "video_type": "other"},
    # {"url": "https://www.youtube.com/watch?v=1sjgYVDrtyU&list=RD1sjgYVDrtyU&start_radio=1", "video_type": "other"},
    # {"url": "https://www.youtube.com/watch?v=yY13X0BKaUw&list=RDyY13X0BKaUw&start_radio=1", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=vaKVbKPQOqY&list=RDvaKVbKPQOqY&start_radio=1", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=81dk4Nc8cFc", "video_type": "fancam"},
    # {"url": "https://www.youtube.com/watch?v=xiifjlrtHfM", "video_type": "fancam"},
    # {"url": "https://www.youtube.com/watch?v=P1ZWb1h2axI", "video_type": "fancam"},
    # {"url": "https://www.youtube.com/watch?v=ErF87xQAQv4&list=RDErF87xQAQv4&start_radio=1", "video_type": "fancam"},
    # {"url": "https://www.youtube.com/watch?v=Q-VX60sGRjo&t=17s", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=7mVKZUAnwVQ&t=3s", "video_type": "fancam"},
    # {"url": "https://www.youtube.com/watch?v=2U2ZAvpzgyY&list=RD2U2ZAvpzgyY&start_radio=1", "video_type": "fancam"},
    # {"url": "https://www.youtube.com/watch?v=WKdkz4x5M90", "video_type": "fancam"},
    # {"url": "https://www.youtube.com/watch?v=zD-AanYwrEE", "video_type": "fancam"},
#]
# ─────────────────────────────────────────────────────────────

# ── video_type 허용값 ─────────────────────────────────────────
# "MV"      — 뮤직비디오 본편
# "teaser"  — 티저 영상
# "live"    — 라이브 클립, 음악방송 무대
# "fancam"  — 직캠
# "shorts"  — 유튜브 쇼츠
# "other"   — 그 외
# ─────────────────────────────────────────────────────────────


# ── 여기에 링크 넣기 (url, video_type만 작성) ─────────────────
VIDEOS = [
    #QWER
    # {"url": "https://www.youtube.com/watch?v=pifz9JH1Re8", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=ImuWa3SJulY", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=AlirzLFEHUI", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=WGm2HmXeeRI", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=On6Pm4M-dQQ", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=JFGRPgYeu38", "video_type": "MV"},
    # {"url": "https://www.youtube.com/watch?v=v6xVldhvlrw", "video_type": "live"},
    # {"url": "https://www.youtube.com/watch?v=Ieb8IOiFfhk", "video_type": "live"},
    # {"url": "https://www.youtube.com/watch?v=BXZKO6fIiAM", "video_type":"live"},
    # {"url": "https://www.youtube.com/watch?v=1KeqWqavpLc", "video_type":"teaser"},
    
    #CNBLUE
    # {"url": "https://www.youtube.com/watch?v=ND5FhEsjVUk&list=RDEMmq3Yb6McBjA4J7BYWGrIlQ&start_radio=1", "video_type":"MV"},
    # {"url": "https://www.youtube.com/watch?v=ai2Afs604_w&list=RDND5FhEsjVUk&index=2", "video_type":"MV"},
    # {"url": "https://www.youtube.com/watch?v=75KBwVtd_W0&list=RDND5FhEsjVUk&index=7", "video_type":"MV"},
    # {"url": "https://www.youtube.com/watch?v=-j86HpSJydY&list=RD-j86HpSJydY&start_radio=1", "video_type":"teaser"},
    # {"url": "https://www.youtube.com/watch?v=m02JpNmb9yk", "video_type":"teaser"},
    
    #FTisland
    {"url": "https://www.youtube.com/watch?v=UgBfFHvXNnY", "video_type":"MV"},
    {"url": "https://www.youtube.com/watch?v=pnrmW_Man7I", "video_type":"MV"},
    {"url": "https://www.youtube.com/watch?v=li5wNSMC5oA&list=RDli5wNSMC5oA&start_radio=1", "video_type":"MV"},
    {"url": "https://www.youtube.com/watch?v=a-Uyxic1tH4&list=RDa-Uyxic1tH4&start_radio=1", "video_type":"teaser"},
    {"url": "https://www.youtube.com/watch?v=uijP_QG8ijs&list=RDuijP_QG8ijs&start_radio=1", "video_type":"live"},
    
]
# ─────────────────────────────────────────────────────────────

def extract_video_id(url):
    url = url.strip()
    match = re.match(r"(?:https?://)?youtu\.be/([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    match = re.match(r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    qs = parse_qs(urlparse(url).query)
    if "v" in qs:
        return qs["v"][0]
    return None


def crawl_video(video_cfg, output_file):
    youtube = build("youtube", "v3", developerKey=API_KEY)

    video_id = extract_video_id(video_cfg["url"])
    if not video_id:
        print(f"[ERROR] 유효하지 않은 링크: {video_cfg['url']}")
        return

    # 영상 정보 가져오기
    info = youtube.videos().list(part="snippet", id=video_id).execute()
    snippet = info["items"][0]["snippet"]
    video_title        = snippet["title"]
    video_published_at = snippet["publishedAt"]

    print(f"[{video_id}] 크롤링 시작 - {video_title}")

    records = []
    next_page_token = None
    crawled_at = datetime.now(timezone.utc).isoformat()

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(MAX_COMMENTS - len(records), 100),
            pageToken=next_page_token,
            textFormat="plainText",
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            records.append({
                "video_id":           video_id,
                "video_type":         video_cfg["video_type"],
                "video_title":        video_title,
                "video_published_at": video_published_at,
                "artist":             ARTIST,
                "purpose":            "train",
                "label":              [],
                "comment_id":         item["snippet"]["topLevelComment"]["id"],
                "text":               comment["textDisplay"],
                "likes":              comment["likeCount"],
                "published_at":       comment["publishedAt"],
                "crawled_at":         crawled_at,
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token or len(records) >= MAX_COMMENTS:
            break

    with open(output_file, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[{video_id}] 완료 - {len(records)}개 댓글 저장")


if __name__ == "__main__":
    for video_cfg in VIDEOS:
        crawl_video(video_cfg, OUTPUT)

    print(f"\n전체 완료 → {OUTPUT}")
