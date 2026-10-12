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
from backend.transcript_service import get_video_transcripts
from backend.rag_service import answer_question as run_rag_question

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


# TEMPORARY integration limit while transcript retrieval/caching is developed
# separately. Once cached transcripts are available, this endpoint can pass all
# cached transcripts into rag_service without changing the RAG implementation.
RAG_TRANSCRIPT_VIDEO_LIMIT = int(os.getenv("RAG_TRANSCRIPT_VIDEO_LIMIT", "3"))

# When True, RAG uses local test transcripts instead of contacting YouTube.
RAG_TEST_MODE = os.getenv(
    "RAG_TEST_MODE",
    "false",
).lower() == "true"

# Location of the local transcript test data.
TEST_TRANSCRIPTS_FILE = BASE_DIR / "data" / "test_transcripts.json"

# Temporary in-memory transcript cache for the RAG branch.
#
# Why this exists:
# The transcript library makes network requests to YouTube. Re-fetching the
# same three transcripts every time a user asks another question is wasteful
# and can increase the chance of YouTube rate/IP blocking. This cache keeps
# successful transcript results for the lifetime of the FastAPI process.
#
# Once the teammate's SQLite transcript cache is merged, this dictionary can
# be removed and the RAG endpoint can read transcripts from SQLite instead.
RAG_TRANSCRIPT_CACHE = {}

# For Testing RAG without transcripts
def load_test_transcripts():
    """Load local transcript data for testing the RAG pipeline."""

    if not TEST_TRANSCRIPTS_FILE.exists():
        raise RuntimeError(
            f"Test transcript file was not found: "
            f"{TEST_TRANSCRIPTS_FILE}"
        )

    with open(
        TEST_TRANSCRIPTS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        transcripts = json.load(file)

    if not isinstance(transcripts, list):
        raise RuntimeError(
            "test_transcripts.json must contain a JSON list."
        )

    return transcripts


def get_rag_transcripts(channel_id, videos):
    """Return either local test transcripts or live YouTube transcripts."""

    # TEST MODE:
    # Do not contact YouTube's transcript service.
    if RAG_TEST_MODE:
        transcripts = load_test_transcripts()
        return transcripts, "test_file"

    # NORMAL MODE:
    # Check the temporary in-memory transcript cache first.
    video_ids = tuple(video["video_id"] for video in videos)
    cache_key = (channel_id, video_ids)

    cached = RAG_TRANSCRIPT_CACHE.get(cache_key)

    if cached is not None:
        return cached, "memory_cache"

    # Only contact YouTube when nothing is cached.
    transcripts = get_video_transcripts(videos)

    if transcripts:
        RAG_TRANSCRIPT_CACHE[cache_key] = transcripts

    return transcripts, "youtube"


@app.post("/api/question")
def ask_question(request: QuestionRequest):
    """Run transcript retrieval + FAISS search + optional Gemini generation.

    This currently uses the existing transcript service as an adapter. The RAG
    module itself is independent of transcript storage, so a teammate can replace
    this adapter with SQLite-cached transcripts later.
    """
    question = request.question.strip()
    channel_id = request.channel_id.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

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

    # Keep transcript work small while the transcript-caching branch is being
    # developed. Change/remove this slice after that work is merged.
    videos = result["videos"][:RAG_TRANSCRIPT_VIDEO_LIMIT]
    transcripts, transcript_source = get_rag_transcripts(channel_id, videos)

    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail="No transcripts were available for the videos being indexed.",
        )

    try:
        rag_result = run_rag_question(
            channel_id=channel_id,
            question=question,
            transcripts=transcripts,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error))

    # The prompt is an internal handoff for the future LLM provider, not data the
    # browser needs to display.
    rag_result.pop("rag_prompt", None)

    return {
        **rag_result,
        "transcript_count": len(transcripts),
        "indexed_video_count": len(videos),
        "transcript_source": transcript_source,
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


# FIXME: Transcript Testing (Will edit more later)
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

    # Only uses a few videos while testing transcripts
    videos = result["videos"][:3]

    # Prints the test videos to the console
    print("TEST VIDEOS:")
    for video in videos:
        print(video["video_id"], video["title"])

    # Retrieves transcripts for the test videos
    transcripts = get_video_transcripts(videos)

    return {
        "transcripts": transcripts,
        "transcript_count": len(transcripts),
    }


@app.get("/api/database/stats")
def database_stats():
    """Small testing endpoint showing how many rows are stored locally."""
    return get_database_stats()


@app.get("/")
def home():
    """Serve the main website when the user visits localhost:8000."""
    return FileResponse(FRONTEND_DIR / "index.html")
