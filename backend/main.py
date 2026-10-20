"""Main FastAPI application for the YouTube Channel Analyzer."""

import json # For testing RAG
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from backend.analytics_service import calculate_analytics
from backend.database_service import (
    get_cached_channel_data,
    get_database_stats,
    initialize_database,
    is_cache_fresh,
    save_channel_and_videos,
)

from backend.youtube_service import get_all_videos, resolve_channel_id
from backend.transcript_service import (
    get_video_transcripts,
    get_video_transcripts_with_status,
)
from backend.rag_service import (
    answer_question as run_rag_question,
    enrich_indexed_video_metadata,
    get_indexed_video_ids,
    get_rag_storage_stats,
    index_transcripts,
)

# Create the FastAPI application.
app = FastAPI(title="YouTube Channel Analyzer")

# Locate the frontend folder so FastAPI can serve HTML/CSS/JavaScript.
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Anything under /static comes from the frontend folder.
# Examples: /static/style.css and /static/script.js
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Create the SQLite database/tables when the application starts importing.
# CREATE TABLE IF NOT EXISTS makes this safe to run every time.
initialize_database()

def get_channel_data(channel_input, refresh=False):
    """Return channel/video data from cache when possible, otherwise sync YouTube.

    Normal behavior:
    1. If the channel is cached and less than 60 minutes old, use SQLite.
    2. Otherwise retrieve the full channel from YouTube.
    3. Save/update the result in SQLite.
    4. Return the saved data.

    refresh=True skips the freshness check and forces a YouTube sync.
    """
    channel_id = resolve_channel_id(channel_input)

    if channel_id is None:
        return None

    if not refresh and is_cache_fresh(channel_id):
        cached_result = get_cached_channel_data(channel_id)
        if cached_result is not None:
            cached_result["data_source"] = "database"
            return cached_result

    # Cache is missing/stale, or the caller explicitly requested a refresh.
    result = get_all_videos(channel_id)

    if result is None:
        return None

    # Upsert the channel and videos so repeated analyses do not duplicate rows.
    save_channel_and_videos(
        channel=result["channel"],
        videos=result["videos"],
    )

    # Read it back from SQLite so both cached and newly synced responses use
    # exactly the same structure.
    saved_result = get_cached_channel_data(channel_id)
    saved_result["data_source"] = "youtube"
    return saved_result


@app.get("/api/health")
def health_check():
    """Simple endpoint used by JavaScript to confirm FastAPI is running."""
    return {
        "status": "success",
        "message": "FastAPI backend is connected successfully!",
    }


@app.get("/api/resolve-channel")
def resolve_channel(channel_input: str):
    """Convert a channel ID or YouTube URL into a channel ID."""

    channel_id = resolve_channel_id(channel_input)

    if channel_id is None:
        raise HTTPException(
            status_code=404,
            detail="YouTube channel was not found.",
        )

    return {"channel_id": channel_id}

# Defines the JSON structure sent by the frontend Q&A form.
class QuestionRequest(BaseModel):
    channel_id: str
    question: str


class RagIndexRequest(BaseModel):
    channel_id: str


# Keep the number small while testing live transcript retrieval. Previously
# indexed Chroma videos are reused, so raising the limit later does not
# re-fetch or re-embed videos that are already stored.
RAG_TRANSCRIPT_VIDEO_LIMIT = int(os.getenv("RAG_TRANSCRIPT_VIDEO_LIMIT", "3"))


def select_rag_videos(videos: list[dict]) -> list[dict]:
    """Select videos for RAG. A limit of 0 means use all available videos."""

    if RAG_TRANSCRIPT_VIDEO_LIMIT == 0:
        return videos

    if RAG_TRANSCRIPT_VIDEO_LIMIT < 0:
        raise ValueError("RAG_TRANSCRIPT_VIDEO_LIMIT must be 0 or a positive integer.")

    return videos[:RAG_TRANSCRIPT_VIDEO_LIMIT]

# Test mode keeps RAG development independent from live transcript requests.
RAG_TEST_MODE = os.getenv(
    "RAG_TEST_MODE",
    "false",
).lower() == "true"

TEST_TRANSCRIPTS_FILE = BASE_DIR / "data" / "test_transcripts.json"


def load_test_transcripts():
    """Load local transcript data for RAG testing without contacting YouTube."""
    if not TEST_TRANSCRIPTS_FILE.exists():
        raise RuntimeError(
            f"Test transcript file was not found: {TEST_TRANSCRIPTS_FILE}"
        )

    with open(TEST_TRANSCRIPTS_FILE, "r", encoding="utf-8") as file:
        transcripts = json.load(file)

    if not isinstance(transcripts, list):
        raise RuntimeError("test_transcripts.json must contain a JSON list.")

    return transcripts


def question_requests_latest_video(question: str) -> bool:
    """Return True for questions explicitly asking about the newest video."""
    text = question.lower()
    asks_for_video = "video" in text or "upload" in text
    return asks_for_video and any(
        phrase in text for phrase in ("most recent", "latest", "newest")
    )


def get_latest_video(videos: list[dict]) -> dict | None:
    """Return the newest video using YouTube publication metadata."""
    dated = [video for video in videos if video.get("published_at")]
    if not dated:
        return videos[0] if videos else None
    return max(dated, key=lambda video: str(video.get("published_at")))


def get_missing_rag_videos(channel_id, videos):
    """Return selected videos whose transcript chunks are not in Chroma yet."""
    video_ids = [video["video_id"] for video in videos]
    indexed_video_ids = get_indexed_video_ids(channel_id, video_ids)

    missing_videos = [
        video
        for video in videos
        if video["video_id"] not in indexed_video_ids
    ]

    return missing_videos, indexed_video_ids


def build_transcript_status(
    requested_video_count,
    indexed_video_count,
    batch_status=None,
):
    """Build a frontend-friendly transcript indexing status message."""
    batch_status = batch_status or {}
    blocked = bool(batch_status.get("blocked"))

    if blocked:
        return {
            "severity": "warning",
            "blocked": True,
            "block_type": batch_status.get("block_type"),
            "blocked_video_id": batch_status.get("blocked_video_id"),
            "indexed_video_count": indexed_video_count,
            "requested_video_count": requested_video_count,
            "new_youtube_transcript_count": batch_status.get("youtube_fetch_count", 0),
            "sqlite_cache_count": batch_status.get("sqlite_cache_count", 0),
            "message": (
                "Transcript retrieval was paused because YouTube temporarily "
                "blocked additional transcript requests. "
                f"{indexed_video_count} of {requested_video_count} selected "
                "videos are currently indexed and remain available for RAG. "
                "Previously cached transcripts were kept. Try again later to "
                "continue indexing new videos."
            ),
        }

    if indexed_video_count < requested_video_count:
        return {
            "severity": "info",
            "blocked": False,
            "indexed_video_count": indexed_video_count,
            "requested_video_count": requested_video_count,
            "new_youtube_transcript_count": batch_status.get("youtube_fetch_count", 0),
            "sqlite_cache_count": batch_status.get("sqlite_cache_count", 0),
            "message": (
                f"RAG transcript coverage: {indexed_video_count} of "
                f"{requested_video_count} selected videos are indexed. "
                "Videos without available transcripts were skipped."
            ),
        }

    return {
        "severity": "success",
        "blocked": False,
        "indexed_video_count": indexed_video_count,
        "requested_video_count": requested_video_count,
        "new_youtube_transcript_count": batch_status.get("youtube_fetch_count", 0),
        "sqlite_cache_count": batch_status.get("sqlite_cache_count", 0),
        "message": (
            f"RAG transcript coverage: {indexed_video_count} of "
            f"{requested_video_count} selected videos are indexed."
        ),
    }


@app.post("/api/rag/index")
def index_rag_content(request: RagIndexRequest):
    """Index transcript content for the active channel.

    This is the ONLY normal application route that retrieves missing
    transcripts for RAG. Question answering does not contact YouTube.

    Live lookup order:
    1. Chroma: skip videos that are already indexed.
    2. SQLite transcript cache: reuse raw transcripts when available.
    3. YouTube transcript retrieval: only for content missing from both stores.
    """
    channel_id = request.channel_id.strip()

    if not channel_id:
        raise HTTPException(status_code=400, detail="Channel ID is required.")

    try:
        result = get_channel_data(channel_id)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Channel request failed: {error}",
        )

    if result is None:
        raise HTTPException(status_code=404, detail="YouTube channel was not found.")

    videos = select_rag_videos(result["videos"])

    if RAG_TEST_MODE:
        rag_channel_id = f"test::{channel_id}"
        test_transcripts = load_test_transcripts()
        for item in test_transcripts:
            if isinstance(item, dict):
                item.setdefault("transcript_source", "test_file")

        try:
            index_summary = index_transcripts(
                rag_channel_id,
                test_transcripts,
                channel_title=result["channel"].get("title"),
            )
            stats = get_rag_storage_stats(rag_channel_id)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=500, detail=str(error))

        return {
            "status": "success",
            "rag_test_mode": True,
            "rag_storage": "chroma",
            "requested_video_count": len(videos),
            "indexed_video_count": stats.get("video_count", 0),
            "newly_indexed_video_count": index_summary["indexed_video_count"],
            "newly_indexed_chunk_count": index_summary["indexed_chunk_count"],
            "skipped_already_indexed_video_count": index_summary["skipped_video_count"],
            "transcript_status": {
                "severity": "info",
                "blocked": False,
                "indexed_video_count": stats.get("video_count", 0),
                "requested_video_count": len(videos),
                "message": (
                    "RAG test mode indexed local test transcripts. "
                    "No YouTube transcript requests were made."
                ),
            },
        }

    try:
        # Existing Chroma databases created by earlier versions can gain
        # publication-date/language metadata without re-embedding their chunks.
        enrich_indexed_video_metadata(
            channel_id,
            videos,
            channel_title=result["channel"].get("title"),
        )
        missing_videos, indexed_before = get_missing_rag_videos(channel_id, videos)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))

    transcript_batch = (
        get_video_transcripts_with_status(missing_videos, channel_id)
        if missing_videos
        else {
            "transcripts": [],
            "blocked": False,
            "youtube_fetch_count": 0,
            "sqlite_cache_count": 0,
            "failed_video_count": 0,
            "block_type": None,
            "blocked_video_id": None,
        }
    )

    new_transcripts = transcript_batch["transcripts"]

    try:
        index_summary = index_transcripts(
            channel_id,
            new_transcripts,
            channel_title=result["channel"].get("title"),
        )
        indexed_after = get_indexed_video_ids(
            channel_id,
            [video["video_id"] for video in videos],
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error))

    transcript_status = build_transcript_status(
        requested_video_count=len(videos),
        indexed_video_count=len(indexed_after),
        batch_status=transcript_batch,
    )

    return {
        "status": "success",
        "rag_test_mode": False,
        "rag_storage": "chroma",
        "requested_video_count": len(videos),
        "indexed_video_count": len(indexed_after),
        "missing_video_count_before_request": len(missing_videos),
        "new_transcript_count": len(new_transcripts),
        "newly_indexed_video_count": index_summary["indexed_video_count"],
        "newly_indexed_chunk_count": index_summary["indexed_chunk_count"],
        "skipped_already_indexed_video_count": index_summary["skipped_video_count"],
        "transcript_batch": {
            "youtube_fetch_count": transcript_batch.get("youtube_fetch_count", 0),
            "sqlite_cache_count": transcript_batch.get("sqlite_cache_count", 0),
            "failed_video_count": transcript_batch.get("failed_video_count", 0),
            "blocked": transcript_batch.get("blocked", False),
            "block_type": transcript_batch.get("block_type"),
            "blocked_video_id": transcript_batch.get("blocked_video_id"),
        },
        "transcript_status": transcript_status,
    }


@app.post("/api/question")
def ask_question(request: QuestionRequest):
    """Answer a question using ONLY transcript content already in Chroma.

    This endpoint deliberately does not retrieve transcripts and does not call
    YouTube. If indexing stopped at 14 of 15 videos, users can still ask
    questions over those 14 videos without retrying the missing transcript.
    """
    question = request.question.strip()
    channel_id = request.channel_id.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not channel_id:
        raise HTTPException(status_code=400, detail="Channel ID is required.")

    rag_channel_id = f"test::{channel_id}" if RAG_TEST_MODE else channel_id

    try:
        stats = get_rag_storage_stats(rag_channel_id)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))

    indexed_video_count = int(stats.get("video_count", 0))
    chunk_count = int(stats.get("chunk_count", 0))

    if chunk_count == 0:
        return {
            "answer": (
                "No transcript content has been indexed for this channel yet. "
                "Use 'Index RAG Content' first, then ask your question again."
            ),
            "sources": [],
            "retrieved_chunk_count": 0,
            "mode": "no_indexed_content",
            "generation_provider": None,
            "generation_model": None,
            "generation_warning": None,
            "newly_indexed_video_count": 0,
            "newly_indexed_chunk_count": 0,
            "skipped_already_indexed_video_count": 0,
            "indexed_video_count": 0,
            "transcript_source": "chroma_only",
            "transcript_status": {
                "severity": "info",
                "blocked": False,
                "indexed_video_count": 0,
                "message": (
                    "No RAG transcript content is indexed yet. "
                    "Question answering did not make any YouTube transcript requests."
                ),
            },
            "rag_storage": "chroma",
            "rag_test_mode": RAG_TEST_MODE,
        }

    target_video_id = None
    query_scope = "channel"

    # Semantic similarity alone cannot interpret words such as "latest" as a
    # date comparison. Resolve that intent using the cached YouTube metadata,
    # then restrict Chroma to the actual newest video's transcript chunks.
    if not RAG_TEST_MODE and question_requests_latest_video(question):
        cached_channel = get_cached_channel_data(channel_id)
        latest_video = get_latest_video((cached_channel or {}).get("videos", []))

        if latest_video is not None:
            target_video_id = str(latest_video["video_id"])
            query_scope = "latest_video"
            latest_is_indexed = bool(
                get_indexed_video_ids(channel_id, [target_video_id])
            )

            if not latest_is_indexed:
                return {
                    "answer": (
                        f"The most recent video, '{latest_video.get('title') or 'Unknown video'}', "
                        "has not been indexed for RAG yet. Use 'Index RAG Content' "
                        "to try indexing it before asking this question again."
                    ),
                    "sources": [],
                    "retrieved_chunk_count": 0,
                    "mode": "latest_video_not_indexed",
                    "generation_provider": None,
                    "generation_model": None,
                    "generation_warning": None,
                    "indexed_video_count": indexed_video_count,
                    "transcript_source": "chroma_only",
                    "query_scope": query_scope,
                    "transcript_status": {
                        "severity": "info",
                        "blocked": False,
                        "indexed_video_count": indexed_video_count,
                        "message": (
                            "The newest channel video is not currently indexed. "
                            "No YouTube transcript request was made by this question."
                        ),
                    },
                    "rag_storage": "chroma",
                    "rag_test_mode": RAG_TEST_MODE,
                }

    try:
        rag_result = run_rag_question(
            channel_id=rag_channel_id,
            question=question,
            transcripts=[],
            target_video_id=target_video_id,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error))

    rag_result.pop("rag_prompt", None)

    return {
        **rag_result,
        "indexed_video_count": indexed_video_count,
        "transcript_source": "chroma_only",
        "transcript_status": {
            "severity": "success",
            "blocked": False,
            "indexed_video_count": indexed_video_count,
            "message": (
                f"RAG searched {indexed_video_count} indexed video"
                f"{'s' if indexed_video_count != 1 else ''} in Chroma. "
                "No new transcript requests were made."
            ),
        },
        "rag_storage": "chroma",
        "rag_test_mode": RAG_TEST_MODE,
        "query_scope": query_scope,
    }


@app.get("/api/channel/{channel_id}")
def get_channel(
    channel_id: str,
    refresh: bool = Query(default=False),
):
    """Return basic channel information using SQLite caching."""
    try:
        result = get_channel_data(channel_id, refresh=refresh)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Channel request failed: {error}",
        )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="YouTube channel was not found.",
        )

    return {
        **result["channel"],
        "data_source": result["data_source"],
        "last_updated": result.get("last_updated"),
    }


@app.get("/api/channel/{channel_id}/videos")
def get_channel_videos(
    channel_id: str,
    refresh: bool = Query(default=False),
):
    """Return all saved/public videos, refreshing YouTube only when needed."""
    try:
        result = get_channel_data(channel_id, refresh=refresh)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Channel request failed: {error}",
        )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="YouTube channel was not found.",
        )

    return result


@app.get("/api/channel/{channel_id}/analytics")
def get_channel_analytics(
    channel_id: str,
    refresh: bool = Query(default=False),
):
    """Return channel data, videos, and analytics using the SQLite cache."""
    try:
        result = get_channel_data(channel_id, refresh=refresh)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Channel request failed: {error}",
        )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="YouTube channel was not found.",
        )

    # Analytics are calculated from the videos read from SQLite.
    analytics = calculate_analytics(result["videos"])

    return {
        "channel": result["channel"],
        "retrieved_video_count": result["retrieved_video_count"],
        "analytics": analytics,
        "videos": result["videos"],
        "data_source": result["data_source"],
        "last_updated": result.get("last_updated"),
    }


@app.get("/api/channel/{channel_id}/transcripts")
def get_channel_transcripts(channel_id: str):
    """Return available transcripts for a channel's videos."""

    try:
        result = get_channel_data(channel_id)
    
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))
    
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Channel request failed: {error}",
        )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="YouTube channel was not found.",
        )

    # Uses the same configurable video selection rule as RAG.
    # 0 means all available videos; a positive value means the newest N videos.
    videos = select_rag_videos(result["videos"])

    # Prints the test videos to the console
    print("TEST VIDEOS:")
    for video in videos:
        print(video["video_id"], video["title"])

    # Retrieves transcripts for the test videos
    transcripts = get_video_transcripts(videos, channel_id)

    return {
        "transcripts": transcripts,
        "transcript_count": len(transcripts),
    }


@app.get("/api/rag/stats")
def rag_stats(
    channel_id: str | None = Query(default=None),
    test_mode: bool | None = Query(default=None),
):
    """Show persistent Chroma counts for testing/debugging.

    When test_mode is omitted, the endpoint follows the application's current
    RAG_TEST_MODE setting so the frontend sees the same namespace as Q&A.
    """
    storage_channel_id = channel_id
    use_test_namespace = RAG_TEST_MODE if test_mode is None else test_mode
    if channel_id and use_test_namespace:
        storage_channel_id = f"test::{channel_id}"

    try:
        return get_rag_storage_stats(storage_channel_id)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/database/stats")
def database_stats():
    """Small testing endpoint showing how many rows are stored locally."""
    return get_database_stats()


@app.get("/")
def home():
    """Serve the main website when the user visits localhost:8000."""
    return FileResponse(FRONTEND_DIR / "index.html")
