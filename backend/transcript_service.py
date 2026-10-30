"""Functions for retrieving and caching YouTube video transcripts.

The batch helper in this module is deliberately conservative with live YouTube
requests:

- SQLite is checked first, so cached transcripts never wait.
- A configurable pause is used only between new YouTube transcript requests.
- RequestBlocked / IpBlocked stop the current batch immediately.
- Successful transcripts retrieved before a block remain saved in SQLite.
- Newly retrieved transcripts preserve their language code and caption timing
  segments so RAG chunks can later expose timestamps.

The service never asks YouTube to translate a caption track. When YouTube video
metadata exposes a default language/audio language, that language is preferred.
Otherwise an original available caption track is selected, preferring an
auto-generated track when one exists because it normally follows the video's
spoken language.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from youtube_transcript_api import (
        IpBlocked,
        RequestBlocked,
        YouTubeTranscriptApi,
    )
except ImportError:  # Backward compatibility with older installed versions.
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import IpBlocked, RequestBlocked

from backend.database_service import get_saved_transcript, save_transcript


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

TRANSCRIPT_REQUEST_DELAY_SECONDS = max(
    0.0,
    float(os.getenv("TRANSCRIPT_REQUEST_DELAY_SECONDS", "8")),
)
TRANSCRIPT_REQUEST_JITTER_SECONDS = max(
    0.0,
    float(os.getenv("TRANSCRIPT_REQUEST_JITTER_SECONDS", "2")),
)


def _preferred_language_codes(video: dict[str, Any]) -> list[str]:
    """Return likely native-language codes from YouTube video metadata."""
    values = [
        video.get("default_audio_language"),
        video.get("default_language"),
    ]

    codes: list[str] = []
    for value in values:
        if not value:
            continue

        code = str(value).strip()
        if not code:
            continue

        # YouTube may report values such as en-US while captions use en.
        for candidate in (code, code.split("-")[0]):
            if candidate and candidate not in codes:
                codes.append(candidate)

    return codes


def _select_original_transcript_track(video: dict[str, Any]):
    """Select an available, non-translated caption track for one video."""
    video_id = str(video["video_id"])
    transcript_list = YouTubeTranscriptApi().list(video_id)
    tracks = list(transcript_list)

    if not tracks:
        raise RuntimeError(f"No transcript tracks are available for {video_id}.")

    preferred_codes = _preferred_language_codes(video)
    if preferred_codes:
        for code in preferred_codes:
            matching = [track for track in tracks if track.language_code == code]
            if matching:
                # Prefer a creator-provided track over generated captions when
                # both exist in the same language.
                manual = [track for track in matching if not track.is_generated]
                return manual[0] if manual else matching[0]

    # If YouTube did not expose a default language, an auto-generated track is
    # usually the strongest indicator of the spoken/native language.
    generated = [track for track in tracks if track.is_generated]
    if generated:
        return generated[0]

    # Manual-only videos may expose multiple caption languages. The list order
    # comes from YouTube's available caption tracks; use the first original
    # track rather than translating one ourselves.
    return tracks[0]


def _normalise_segments(fetched_transcript) -> list[dict[str, Any]]:
    """Return JSON-safe caption segments with start/duration/end timestamps."""
    segments: list[dict[str, Any]] = []

    for snippet in fetched_transcript:
        start = float(snippet.start)
        duration = float(snippet.duration)
        segments.append(
            {
                "text": str(snippet.text),
                "start": start,
                "duration": duration,
                "end": start + duration,
            }
        )

    return segments


def _fetch_and_save_youtube_transcript(
    video: dict[str, Any],
    channel_id: str,
) -> dict[str, Any]:
    """Retrieve one original-language transcript and persist it in SQLite."""
    video_id = str(video["video_id"])
    print(f"Fetching transcript from YouTube: {video_id}")

    track = _select_original_transcript_track(video)
    fetched = track.fetch()

    segments = _normalise_segments(fetched)
    transcript_text = " ".join(segment["text"] for segment in segments)
    language_code = str(getattr(fetched, "language_code", "") or "").strip() or None

    # Persist immediately so a later block does not lose successful work.
    save_transcript(
        video_id,
        channel_id,
        transcript_text,
        language=language_code,
        segments=segments,
    )

    return {
        "video_id": video_id,
        "title": str(video.get("title") or "Unknown video"),
        "published_at": video.get("published_at"),
        "duration_seconds": video.get("duration_seconds"),
        "language": language_code,
        "transcript_source": "youtube_transcript_api",
        "transcript": transcript_text,
        "segments": segments,
    }


def _decode_saved_segments(saved: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode timestamp segments from an existing SQLite transcript row."""
    raw = saved.get("segments_json")
    if not raw:
        return []

    try:
        segments = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []

    return segments if isinstance(segments, list) else []


def _saved_transcript_record(
    saved: dict[str, Any],
    video: dict[str, Any],
) -> dict[str, Any]:
    """Convert a SQLite transcript row to the structure consumed by RAG."""
    return {
        "video_id": str(video["video_id"]),
        "title": str(video.get("title") or "Unknown video"),
        "published_at": video.get("published_at"),
        "duration_seconds": video.get("duration_seconds"),
        "language": saved.get("language")
        or video.get("default_audio_language")
        or video.get("default_language"),
        "transcript_source": "sqlite_cache",
        "transcript": saved["transcript"],
        "segments": _decode_saved_segments(saved),
    }


def get_video_transcript(video_id: str, channel_id: str) -> str:
    """Backward-compatible single-video helper returning transcript text."""
    saved = get_saved_transcript(video_id)

    if saved is not None:
        print(f"Using cached transcript from SQLite: {video_id}")
        return saved["transcript"]

    record = _fetch_and_save_youtube_transcript(
        {"video_id": video_id, "title": "Unknown video"},
        channel_id,
    )
    return record["transcript"]


def _delay_before_next_youtube_request() -> float:
    """Pause before another live YouTube transcript request and return seconds."""
    jitter = (
        random.uniform(0.0, TRANSCRIPT_REQUEST_JITTER_SECONDS)
        if TRANSCRIPT_REQUEST_JITTER_SECONDS > 0
        else 0.0
    )
    delay = TRANSCRIPT_REQUEST_DELAY_SECONDS + jitter

    if delay > 0:
        print(f"Waiting {delay:.1f}s before the next YouTube transcript request...")
        time.sleep(delay)

    return delay


def get_video_transcripts_with_status(
    videos: list[dict[str, Any]],
    channel_id: str,
    should_stop=None,
    on_transcript=None,
    on_progress=None,
) -> dict[str, Any]:
    """Retrieve/cache a transcript batch and return progress/block information.

    Optional callbacks are used by the web indexing job so status can update
    while a long batch is running. Existing callers can ignore them.
    """
    transcripts: list[dict[str, Any]] = []

    sqlite_cache_count = 0
    youtube_fetch_count = 0
    failed_video_count = 0
    youtube_request_count = 0

    blocked = False
    cancelled = False
    block_type = None
    blocked_video_id = None
    blocked_error = None
    has_made_youtube_request = False

    for position, video in enumerate(videos, start=1):
        if should_stop and should_stop():
            cancelled = True
            break

        video_id = str(video["video_id"])

        if on_progress:
            on_progress({
                "event": "starting",
                "position": position,
                "total": len(videos),
                "video_id": video_id,
                "title": video.get("title"),
            })

        # Cache lookup happens before any delay. Cached transcripts should stay
        # fast and should not count as YouTube requests.
        saved = get_saved_transcript(video_id)
        if saved is not None:
            print(f"Using cached transcript from SQLite: {video_id}")
            sqlite_cache_count += 1
            record = _saved_transcript_record(saved, video)
            transcripts.append(record)
            if on_transcript:
                on_transcript(record, {"source": "sqlite_cache", "position": position, "total": len(videos)})
            if on_progress:
                on_progress({"event": "transcript_ready", "source": "sqlite_cache", "position": position, "total": len(videos), "video_id": video_id, "title": video.get("title")})
            continue

        if has_made_youtube_request:
            _delay_before_next_youtube_request()

        has_made_youtube_request = True
        youtube_request_count += 1

        try:
            record = _fetch_and_save_youtube_transcript(video, channel_id)
            youtube_fetch_count += 1
            transcripts.append(record)
            if on_transcript:
                on_transcript(record, {"source": "youtube_transcript_api", "position": position, "total": len(videos)})
            if on_progress:
                on_progress({"event": "transcript_ready", "source": "youtube_transcript_api", "position": position, "total": len(videos), "video_id": video_id, "title": video.get("title")})

        except (RequestBlocked, IpBlocked) as error:
            blocked = True
            block_type = type(error).__name__
            blocked_video_id = video_id
            blocked_error = str(error)

            print(
                "Transcript retrieval stopped because YouTube blocked the "
                f"request ({block_type}) for {video_id}."
            )
            if on_progress:
                on_progress({"event": "blocked", "position": position, "total": len(videos), "video_id": video_id, "title": video.get("title"), "block_type": block_type})
            break

        except Exception as error:
            failed_video_count += 1
            print(f"Transcript error for {video_id}: {error}")
            if on_progress:
                on_progress({"event": "failed", "position": position, "total": len(videos), "video_id": video_id, "title": video.get("title"), "error": str(error)})
            continue

    successful_count = len(transcripts)
    remaining_video_count = max(
        0,
        len(videos) - successful_count - failed_video_count,
    )

    return {
        "transcripts": transcripts,
        "requested_video_count": len(videos),
        "successful_transcript_count": successful_count,
        "sqlite_cache_count": sqlite_cache_count,
        "youtube_fetch_count": youtube_fetch_count,
        "youtube_request_count": youtube_request_count,
        "failed_video_count": failed_video_count,
        "remaining_video_count": remaining_video_count,
        "blocked": blocked,
        "cancelled": cancelled,
        "block_type": block_type,
        "blocked_video_id": blocked_video_id,
        "blocked_error": blocked_error,
    }


def get_video_transcripts(videos, channel_id):
    """Backward-compatible helper returning only the transcript list."""
    return get_video_transcripts_with_status(videos, channel_id)["transcripts"]


def get_cached_video_transcripts(videos: list[dict[str, Any]]) -> dict[str, Any]:
    """Return transcript records from SQLite only, without contacting YouTube."""
    transcripts: list[dict[str, Any]] = []
    missing_video_ids: list[str] = []

    for video in videos:
        video_id = str(video.get("video_id") or "")
        if not video_id:
            continue

        saved = get_saved_transcript(video_id)
        if saved is None:
            missing_video_ids.append(video_id)
            continue

        transcripts.append(_saved_transcript_record(saved, video))

    return {
        "transcripts": transcripts,
        "cached_transcript_count": len(transcripts),
        "missing_video_ids": missing_video_ids,
    }


if __name__ == "__main__":
    from backend.youtube_service import get_all_videos

    channel_id = "UCBNvuaTVFUfQlVt9uLeMOfA"
    result = get_all_videos(channel_id)
    videos = result["videos"][:1]
    batch = get_video_transcripts_with_status(videos, channel_id)

    for item in batch["transcripts"]:
        print("\nVIDEO:", item["title"])
        print("LANGUAGE:", item.get("language"))
        print("SEGMENTS:", len(item.get("segments") or []))
        print(item["transcript"][:500])

    print("\nBatch status:", {k: v for k, v in batch.items() if k != "transcripts"})
