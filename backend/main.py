"""Main FastAPI application for the YouTube Channel Analyzer."""

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

# Defines the JSON structure for the temporary Q&A endpoint.
class QuestionRequest(BaseModel):
    question: str


@app.post("/api/question")
def ask_question(request: QuestionRequest):
    """Temporary endpoint; the real RAG pipeline will replace this later."""
    return {
        "answer": "API: RAG answer will be connected here."
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
