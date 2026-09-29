"""Main FastAPI application for the YouTube Channel Analyzer."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from backend.analytics_service import calculate_analytics
from backend.youtube_service import get_channel_by_id, get_recent_videos

# Create the FastAPI application.
app = FastAPI(title="YouTube Channel Analyzer")

# Locate the frontend folder so FastAPI can serve HTML/CSS/JavaScript.
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Anything under /static comes from the frontend folder.
# Examples: /static/style.css and /static/script.js
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/api/health")
def health_check():
    """Simple endpoint used by JavaScript to confirm FastAPI is running."""
    return {
        "status": "success",
        "message": "FastAPI backend is connected successfully!",
    }

# Defines the JSON structure
class QuestionRequest(BaseModel):
    question: str

# Receives a question from the frontend and returns a temporary answer
@app.post("/api/question")
def ask_question(request: QuestionRequest):
    return {
        "answer": "API: RAG answer will be connected here."
    }

@app.get("/api/channel/{channel_id}")
def get_channel(channel_id: str):
    """Return basic information for one channel."""
    try:
        channel = get_channel_by_id(channel_id)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"YouTube API request failed: {error}")

    if channel is None:
        raise HTTPException(status_code=404, detail="YouTube channel was not found.")

    return channel


@app.get("/api/channel/{channel_id}/videos")
def get_channel_videos(
    channel_id: str,
    # FastAPI validates that limit stays between 1 and 50.
    limit: int = Query(default=10, ge=1, le=50),
):
    """Return recent videos for a channel."""
    try:
        result = get_recent_videos(channel_id, limit)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"YouTube API request failed: {error}")

    if result is None:
        raise HTTPException(status_code=404, detail="YouTube channel was not found.")

    return result


@app.get("/api/channel/{channel_id}/analytics")
def get_channel_analytics(
    channel_id: str,
    limit: int = Query(default=10, ge=1, le=50),
):
    """Return channel data, recent videos, and calculated analytics."""
    try:
        # Retrieve the raw YouTube video data first.
        result = get_recent_videos(channel_id, limit)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"YouTube API request failed: {error}")

    if result is None:
        raise HTTPException(status_code=404, detail="YouTube channel was not found.")

    # Send the retrieved videos to our separate analytics layer.
    analytics = calculate_analytics(result["videos"])

    # The frontend receives everything it needs in one response.
    return {
        "channel": result["channel"],
        "retrieved_video_count": result["retrieved_video_count"],
        "analytics": analytics,
        "videos": result["videos"],
    }


@app.get("/")
def home():
    """Serve the main website when the user visits localhost:8000."""
    return FileResponse(FRONTEND_DIR / "index.html")
