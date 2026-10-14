#!/usr/bin/env python3
"""Download captions for every video on a YouTube channel into a local folder.

    python fetch_captions.py --out ./transcripts            # whole channel
    python fetch_captions.py --out ./transcripts --limit 3  # quick test

Resumable: videos already saved (or recorded in failed.log) are skipped.
For personal study use; respect YouTube's terms and creators' rights.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import yt_dlp

from vtt_clean import clean_vtt

DEFAULT_CHANNEL = "https://www.youtube.com/@InnerCircleTrader/videos"


def slugify(title: str, n: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:n] or "untitled"


def list_videos(channel: str, limit: int | None) -> list[dict]:
    opts = {"extract_flat": True, "quiet": True, "skip_download": True}
    if limit:
        opts["playlistend"] = limit
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel, download=False)
    return [e for e in info.get("entries", []) if e and e.get("id")]


def fetch_one(video_id: str, tmp: Path, lang: str) -> tuple[dict, str | None]:
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": str(tmp / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    vtts = sorted(tmp.glob(f"{video_id}*.vtt"))
    if not vtts:
        return info, None
    text = clean_vtt(vtts[0].read_text(encoding="utf-8", errors="ignore"))
    for v in vtts:
        v.unlink()
    return info, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default=DEFAULT_CHANNEL)
    ap.add_argument("--out", type=Path, default=Path("transcripts"))
    ap.add_argument("--lang", default="en")
    ap.add_argument("--limit", type=int, help="only the newest N videos")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between videos")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / ".tmp"
    tmp.mkdir(exist_ok=True)
    failed_log = args.out / "failed.log"
    failed = set(failed_log.read_text().split()) if failed_log.exists() else set()
    done = {p.name.split("_")[1] for p in args.out.glob("*_*_*.txt")}

    videos = list_videos(args.channel, args.limit)
    print(f"{len(videos)} videos found; {len(done)} already saved")
    saved = 0
    for i, v in enumerate(videos, 1):
        vid = v["id"]
        if vid in done or vid in failed:
            continue
        try:
            info, text = fetch_one(vid, tmp, args.lang)
        except Exception as e:  # network/rate-limit/private video
            print(f"[{i}/{len(videos)}] {vid} error: {e}", file=sys.stderr)
            time.sleep(args.sleep * 5)
            continue
        if not text:
            with failed_log.open("a") as f:
                f.write(vid + "\n")
            print(f"[{i}/{len(videos)}] {vid} no captions")
        else:
            date = info.get("upload_date") or "00000000"
            title = info.get("title") or v.get("title") or vid
            path = args.out / f"{date}_{vid}_{slugify(title)}.txt"
            path.write_text(
                f"TITLE: {title}\nURL: https://www.youtube.com/watch?v={vid}\nDATE: {date}\n\n{text}\n",
                encoding="utf-8",
            )
            saved += 1
            print(f"[{i}/{len(videos)}] saved {path.name}")
        time.sleep(args.sleep)
    print(f"Done. {saved} new transcripts in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
