"""Standalone YouTube transcript preprocessing utility.

"Offline utility" in this project means the transcript preprocessing step runs
separately from the web UI. This command may still use the YouTube Data API and
youtube-transcript-api to obtain public channel/video/caption data.

Run from the project root, for example:
    python youtube_transcriptor.py UCxxxxxxxx --limit 10
    python youtube_transcriptor.py https://youtube.com/@handle --limit 0

A limit of 0 means all available channel videos. Successfully retrieved
transcripts are saved immediately to SQLite so a later block does not lose work.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from backend.database_service import (
    get_cached_channel_data,
    initialize_database,
    is_cache_fresh,
    save_channel_and_videos,
)
from backend.transcript_service import get_video_transcripts_with_status
from backend.youtube_service import get_all_videos, resolve_channel_id


load_dotenv()


def _select_videos(videos: list[dict], limit: int) -> list[dict]:
    if limit < 0:
        raise ValueError("--limit must be 0 or a positive integer.")
    return videos if limit == 0 else videos[:limit]


def _load_channel(channel_input: str, refresh: bool) -> tuple[str, dict]:
    channel_id = resolve_channel_id(channel_input)
    if not channel_id:
        raise RuntimeError("YouTube channel could not be resolved.")

    if not refresh and is_cache_fresh(channel_id):
        cached = get_cached_channel_data(channel_id)
        if cached is not None:
            return channel_id, cached

    result = get_all_videos(channel_id)
    if result is None:
        raise RuntimeError("YouTube channel was not found.")

    save_channel_and_videos(result["channel"], result["videos"])
    cached = get_cached_channel_data(channel_id)
    if cached is None:
        raise RuntimeError("Channel metadata could not be saved to SQLite.")
    return channel_id, cached


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve native/original-language YouTube transcripts into SQLite."
    )
    parser.add_argument("channel", help="YouTube UC channel ID or supported channel URL")
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("RAG_TRANSCRIPT_VIDEO_LIMIT", "0")),
        help="Newest N videos to process; 0 means all (default: env value or 0).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh channel/video metadata from YouTube before processing.",
    )
    args = parser.parse_args()

    initialize_database()

    try:
        channel_id, channel_data = _load_channel(args.channel, args.refresh)
        videos = _select_videos(channel_data["videos"], args.limit)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Channel: {channel_data['channel'].get('title') or channel_id}")
    print(f"Channel ID: {channel_id}")
    print(f"Videos selected: {len(videos)}")

    batch = get_video_transcripts_with_status(videos, channel_id)
    transcripts = batch["transcripts"]
    languages = sorted(
        {str(item.get("language")) for item in transcripts if item.get("language")}
    )

    print("\nTranscript preprocessing complete")
    print(f"  Successful/cached transcripts: {batch['successful_transcript_count']}")
    print(f"  Reused from SQLite: {batch['sqlite_cache_count']}")
    print(f"  Newly fetched from YouTube: {batch['youtube_fetch_count']}")
    print(f"  Failed videos: {batch['failed_video_count']}")
    print(f"  Languages observed: {', '.join(languages) if languages else 'unknown'}")

    if batch.get("blocked"):
        print(
            f"  YouTube block: {batch.get('block_type')} on "
            f"{batch.get('blocked_video_id')}. Successful work was preserved."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
