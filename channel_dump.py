#!/usr/bin/env python3
"""
Fetch video metadata from a YouTube channel and download available transcripts.
Outputs:
- data/raw/manifest.csv           (video metadata)
- data/raw/transcripts.jsonl      (one JSON per line with transcript text + metadata)
- data/raw/transcripts/{id}.json  (per-video full record)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import yt_dlp


# ----------------------------
# Data models
# ----------------------------
@dataclass
class VideoMeta:
    id: str
    title: str
    url: str
    uploader: Optional[str]
    upload_date: Optional[str]
    duration: Optional[int]  # seconds
    view_count: Optional[int]


@dataclass
class TranscriptRecord:
    meta: VideoMeta
    language: Optional[str]
    is_generated: Optional[bool]
    text: str


# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs(base: Path) -> Dict[str, Path]:
    out_raw = base / "data" / "raw"
    out_tx_dir = out_raw / "transcripts"
    out_tx_dir.mkdir(parents=True, exist_ok=True)
    return {
        "raw": out_raw,
        "tx_dir": out_tx_dir,
        "manifest_csv": out_raw / "manifest.csv",
        "transcripts_jsonl": out_raw / "transcripts.jsonl",
    }


def _ydl() -> yt_dlp.YoutubeDL:
    # extract_flat avoids fetching formats; faster for listing
    opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "nocheckcertificate": True,
        "skip_download": True,
        "dump_single_json": True,
    }
    return yt_dlp.YoutubeDL(opts)


def list_channel_videos(channel_url: str, max_videos: Optional[int] = None) -> List[VideoMeta]:
    """
    Accepts channel URLs like:
      - https://www.youtube.com/@handle
      - https://www.youtube.com/c/Name
      - https://www.youtube.com/channel/UCxxxx
      - Any playlist (e.g., uploads)
    """
    ydl = _ydl()
    info = ydl.extract_info(channel_url, download=False)

    # --- normalize & clean ---
    if isinstance(info, dict) and ("entries" in info):
        # channel/playlist case; entries might be [] or contain junk
        entries = info["entries"] or []
    else:
        # single-video case (or defensive fallback)
        entries = [info] if info is not None else []

    # Clean: keep only dicts that actually have an 'id'
    entries = [e for e in entries if isinstance(e, dict) and e.get("id")]

    videos: List[VideoMeta] = []

    for e in entries:
        # Some entries are shallow (flat) dicts; use .get defensively
        vid = e.get("id")
        if not vid:
            continue
        url = f"https://www.youtube.com/watch?v={vid}"
        videos.append(
            VideoMeta(
                id=vid,
                title=e.get("title"),
                url=url,
                uploader=e.get("uploader"),
                upload_date=e.get("upload_date"),  # YYYYMMDD or None
                duration=e.get("duration"),        # seconds or None
                view_count=e.get("view_count"),
            )
        )
        if max_videos and len(videos) >= max_videos:
            break

    return videos


def pick_transcript_variant(video_id: str, preferred_langs: list[str]) -> dict | None:
    """
    Returns dict { 'lang', 'is_generated', 'segments' } where 'segments' is a list[dict].
    Works with youtube-transcript-api >=1.2.x (new) and older releases (legacy).
    """
    # Try the new instance API first (v1.2+)
    try:
        api = YouTubeTranscriptApi()  # new API uses an instance with .list() / .fetch()
        has_new_api = hasattr(api, "list") and hasattr(api, "fetch")
    except TypeError:
        has_new_api = False

    if has_new_api:
        try:
            tx_list = api.list(video_id)
        except Exception:
            return None

        # Prefer manually created captions in preferred languages
        for lang in preferred_langs:
            try:
                t = tx_list.find_manually_created_transcript([lang])
                raw = t.fetch().to_raw_data()  # normalize to list[dict]
                return {"lang": t.language_code, "is_generated": False, "segments": raw}
            except Exception:
                pass

        # Then auto-generated
        for lang in preferred_langs:
            try:
                t = tx_list.find_generated_transcript([lang])
                raw = t.fetch().to_raw_data()
                return {"lang": t.language_code, "is_generated": True, "segments": raw}
            except Exception:
                pass

        # Fallback: first available
        try:
            t = next(iter(tx_list))
            raw = t.fetch().to_raw_data()
            return {"lang": t.language_code, "is_generated": t.is_generated, "segments": raw}
        except Exception:
            return None

    # ----- Legacy API path (pre-1.2) -----
    # Falls back to class methods list_transcripts / get_transcript if available.
    list_fn = getattr(YouTubeTranscriptApi, "list_transcripts", None)
    if callable(list_fn):
        try:
            tx_list = list_fn(video_id)
        except Exception:
            return None

        for lang in preferred_langs:
            try:
                t = tx_list.find_manually_created_transcript([lang])
                return {"lang": t.language_code, "is_generated": False, "segments": t.fetch()}
            except Exception:
                pass
        for lang in preferred_langs:
            try:
                t = tx_list.find_generated_transcript([lang])
                return {"lang": t.language_code, "is_generated": True, "segments": t.fetch()}
            except Exception:
                pass
        try:
            t = next(iter(tx_list))
            return {"lang": t.language_code, "is_generated": t.is_generated, "segments": t.fetch()}
        except Exception:
            return None

    # Last resort: very old get_transcript signature
    for lang in preferred_langs:
        try:
            segs = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            return {"lang": lang, "is_generated": None, "segments": segs}
        except Exception:
            pass
    try:
        segs = YouTubeTranscriptApi.get_transcript(video_id)
        return {"lang": None, "is_generated": None, "segments": segs}
    except Exception:
        return None

def segments_to_text(segments: List[Dict]) -> str:
    # Concatenate lines; strip extra whitespace
    lines = []
    for s in segments:
        txt = (s.get("text") or "").strip()
        if txt:
            lines.append(txt)
    return "\n".join(lines)


def write_manifest(path_csv: Path, videos: List[VideoMeta]) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "url", "uploader", "upload_date", "duration_sec", "view_count"])
        for v in videos:
            w.writerow([v.id, v.title, v.url, v.uploader, v.upload_date, v.duration, v.view_count])


def append_jsonl(path_jsonl: Path, obj: Dict) -> None:
    with path_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------
# Main pipeline
# ----------------------------
def fetch_and_store(
    channel_url: str,
    out_base: Path,
    preferred_langs: List[str],
    max_videos: Optional[int] = None,
    delay_s: float = 0.4,
) -> None:
    paths = ensure_dirs(out_base)

    print(f"[1/3] Listing videos from: {channel_url}")
    videos = list_channel_videos(channel_url, max_videos=max_videos)
    print(f"  → Found {len(videos)} videos")

    print(f"[2/3] Writing manifest: {paths['manifest_csv']}")
    write_manifest(paths["manifest_csv"], videos)

    print(f"[3/3] Fetching transcripts into: {paths['tx_dir']}")
    for i, v in enumerate(videos, start=1):
        print(f"  ({i}/{len(videos)}) {v.id}  {v.title!r}")
        rec_path = paths["tx_dir"] / f"{v.id}.json"
        if rec_path.exists():
            print("    - already exists, skipping")
            continue

        variant = pick_transcript_variant(v.id, preferred_langs)
        if not variant:
            print("    - no transcript available")
            continue

        text = segments_to_text(variant["segments"])
        record = TranscriptRecord(
            meta=v,
            language=variant.get("lang"),
            is_generated=variant.get("is_generated"),
            text=text,
        )
        payload = {
            "meta": asdict(record.meta),
            "language": record.language,
            "is_generated": record.is_generated,
            "text": record.text,
        }

        # per-video JSON
        with rec_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # append to combined JSONL
        append_jsonl(paths["transcripts_jsonl"], payload)

        # polite delay (avoid hammering APIs)
        time.sleep(delay_s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump YouTube channel videos + transcripts")
    p.add_argument("channel_url", help="YouTube channel or playlist URL (e.g., https://www.youtube.com/@handle)")
    p.add_argument("--out", default=".", help="Project root (default: current dir)")
    p.add_argument("--langs", default="en,en-US,en-GB", help="Preferred languages (comma-separated)")
    p.add_argument("--max", type=int, default=None, help="Max number of videos to process (default: all)")
    p.add_argument("--delay", type=float, default=0.4, help="Delay between transcript fetches (seconds)")
    return p.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.out).resolve()
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    fetch_and_store(
        channel_url=args.channel_url,
        out_base=out_base,
        preferred_langs=langs,
        max_videos=args.max,
        delay_s=args.delay,
    )


if __name__ == "__main__":
    main()
