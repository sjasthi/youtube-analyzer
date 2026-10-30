"""Main FastAPI application for the YouTube Channel Analyzer."""

import json # For testing RAG
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from backend.analytics_service import calculate_analytics
from backend.database_service import (
    get_cached_channel_data,
    get_database_stats,
    get_saved_transcripts_for_channel,
    get_saved_transcript,
    get_saved_transcript_with_video,
    initialize_database,
    is_cache_fresh,
    save_channel_and_videos,
)

from backend.youtube_service import get_all_videos, resolve_channel_id
from backend.llm_service import generate_with_gemini
from backend.transcript_service import (
    get_cached_video_transcripts,
    get_video_transcripts,
    get_video_transcripts_with_status,
)
from backend.rag_service import (
    EMBEDDING_MODEL_NAME,
    answer_question as run_rag_question,
    channel_requires_embedding_rebuild,
    delete_channel_index,
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
    answer_language: str = Field(default="auto", max_length=80)
    video_id: str | None = None
    session_id: str | None = Field(default=None, max_length=120)


class RagIndexRequest(BaseModel):
    channel_id: str
    video_id: str | None = None


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


def _normalise_title_text(value: str) -> str:
    """Normalise user/title text for conservative title matching."""
    return " ".join(re.findall(r"[\w]+", str(value or "").lower(), flags=re.UNICODE))


def find_requested_video(question: str, videos: list[dict]) -> dict | None:
    """Find a video explicitly named in a question.

    Matching is deliberately conservative: a full title must appear in the
    question, or a quoted phrase must clearly match part of one title. This
    avoids incorrectly forcing ordinary topic questions onto a random video.
    """
    clean_question = str(question or "")
    normalised_question = _normalise_title_text(clean_question)

    # Direct video IDs are unambiguous.
    for video in videos:
        video_id = str(video.get("video_id") or "")
        if len(video_id) >= 8 and video_id in clean_question:
            return video

    exact_matches: list[tuple[int, dict]] = []
    for video in videos:
        title = str(video.get("title") or "").strip()
        normalised_title = _normalise_title_text(title)
        if len(normalised_title) >= 4 and normalised_title in normalised_question:
            exact_matches.append((len(normalised_title), video))

    if exact_matches:
        return max(exact_matches, key=lambda item: item[0])[1]

    # Quoted partial titles such as: What is "Python Basics" about?
    quoted_phrases = re.findall(r'["“”]([^"“”]{3,})["“”]', clean_question)
    for phrase in quoted_phrases:
        normalised_phrase = _normalise_title_text(phrase)
        if not normalised_phrase:
            continue
        candidates = []
        for video in videos:
            title = str(video.get("title") or "")
            normalised_title = _normalise_title_text(title)
            if normalised_phrase in normalised_title or normalised_title in normalised_phrase:
                candidates.append((len(normalised_title), video))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]

    return None


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




_ANALYTICS_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _analytics_rank_request(question: str) -> tuple[int, str, str] | None:
    """Detect ranking questions that should use cached analytics metadata.

    Examples:
    - "What is the most liked video?"
    - "Which video has the most views?"
    - "Top 5 most commented videos"
    - "Which video has the fewest likes?"

    Returns (count, SQLite field name, direction) or None.
    """
    text = " ".join(str(question or "").lower().split())
    if not text or not any(word in text for word in ("video", "videos", "upload", "uploads")):
        return None

    metric = None
    if re.search(r"\blike(?:s|d)?\b", text):
        metric = "like_count"
    elif re.search(r"\bview(?:s|ed)?\b", text):
        metric = "view_count"
    elif re.search(r"\bcomment(?:s|ed)?\b", text):
        metric = "comment_count"
    if metric is None:
        return None

    direction = None
    if re.search(r"\b(?:most|highest|top|greatest)\b", text):
        direction = "desc"
    elif re.search(r"\b(?:least|fewest|lowest)\b", text):
        direction = "asc"
    if direction is None:
        return None

    count = 1
    numeric = re.search(r"\btop\s+(\d{1,3})\b", text)
    if numeric:
        count = max(1, int(numeric.group(1)))
    else:
        word_match = re.search(
            r"\btop\s+(" + "|".join(_ANALYTICS_NUMBER_WORDS) + r")\b",
            text,
        )
        if word_match:
            count = _ANALYTICS_NUMBER_WORDS[word_match.group(1)]

    return count, metric, direction


def _explicit_response_language(question: str) -> str | None:
    """Return a natural-language output request such as 'Answer in Japanese'."""
    text = str(question or "").strip()
    patterns = (
        r"(?:answer|respond|reply)\s+(?:to\s+me\s+)?in\s+([A-Za-z][A-Za-z ()-]{1,50}?)(?:[.!?,]|$)",
        r"translate(?:\s+the\s+answer)?\s+(?:to|into)\s+([A-Za-z][A-Za-z ()-]{1,50}?)(?:[.!?,]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _analytics_answer_language(answer_language: str, question: str) -> str | None:
    requested = str(answer_language or "").strip()
    if requested and requested.lower() != "auto":
        return requested
    return _explicit_response_language(question)


def _translate_metadata_answer(answer: str, language: str | None) -> str:
    """Translate a deterministic metadata answer only when the user requests it.

    Analytics facts themselves remain computed locally from SQLite. Gemini is
    used only for the final wording when a different output language is
    explicitly requested.
    """
    if not language:
        return answer
    prompt = (
        f"Translate the following factual answer into {language}. Preserve video titles, "
        "numbers, punctuation, and ranking order exactly. Do not add, remove, or infer facts. "
        "Return only the translated answer.\n\n"
        f"Answer:\n{answer}"
    )
    try:
        return str(generate_with_gemini(prompt).get("answer") or answer).strip() or answer
    except RuntimeError:
        # The analytics result is still useful even when Gemini is unavailable.
        return answer


def _analytics_metric_label(metric: str) -> tuple[str, str]:
    labels = {
        "like_count": ("liked", "likes"),
        "view_count": ("viewed", "views"),
        "comment_count": ("commented", "comments"),
    }
    return labels[metric]


def _answer_analytics_rank_query(
    question: str,
    videos: list[dict],
    request: "QuestionRequest",
    session_id: str,
    indexed_video_count: int,
) -> dict | None:
    """Answer likes/views/comments ranking questions from SQLite metadata."""
    parsed = _analytics_rank_request(question)
    # Ranking/identity questions are answered directly from SQLite. If the user
    # asks what the ranked video says/discusses/is about, let the metadata router
    # select the correct video and continue into transcript RAG instead.
    if parsed is None or not videos or _requests_video_content(question):
        return None

    requested_count, metric, direction = parsed
    reverse = direction == "desc"
    ranked = sorted(
        videos,
        key=lambda video: int(video.get(metric) or 0),
        reverse=reverse,
    )
    selected = ranked[:requested_count]
    if not selected:
        return None

    adjective, noun = _analytics_metric_label(metric)
    if requested_count == 1:
        video = selected[0]
        value = int(video.get(metric) or 0)
        polarity = "most" if reverse else "least"
        answer = (
            f'The {polarity} {adjective} video is "{video.get("title") or video.get("video_id")}" '
            f"with {value:,} {noun}."
        )
    else:
        polarity = "most" if reverse else "least"
        lines = [
            f'{index}. {video.get("title") or video.get("video_id")} — {int(video.get(metric) or 0):,} {noun}'
            for index, video in enumerate(selected, start=1)
        ]
        answer = f"Top {len(selected)} {polarity} {adjective} videos:\n" + "\n".join(lines)
        if len(selected) < requested_count:
            answer += f"\n\nOnly {len(selected)} cached video(s) are available out of the {requested_count} requested."

    requested_language = _analytics_answer_language(request.answer_language, question)
    answer = _translate_metadata_answer(answer, requested_language)
    selected_ids = [str(video.get("video_id") or "") for video in selected]
    sources = [
        {
            "video_id": str(video.get("video_id") or ""),
            "title": str(video.get("title") or video.get("video_id") or "Untitled video"),
            "video_url": f'https://www.youtube.com/watch?v={str(video.get("video_id") or "")}',
            "published_at": video.get("published_at"),
            "duration_seconds": video.get("duration_seconds"),
            "language": video.get("default_audio_language") or video.get("default_language"),
            "start_time": None,
            "end_time": None,
        }
        for video in selected
    ]

    with _CHAT_LOCK:
        active = _CHAT_SESSIONS.get(session_id)
        if active is not None:
            active["messages"].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            active["messages"] = active["messages"][-_MAX_CHAT_MESSAGES:]
            active["target_video_ids"] = selected_ids
            active["updated_at"] = datetime.now(timezone.utc).isoformat()

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunk_count": 0,
        "mode": "analytics_metadata",
        "indexed_video_count": indexed_video_count,
        "rag_storage": "chroma",
        "rag_test_mode": RAG_TEST_MODE,
        "query_scope": f"analytics_{metric}_{direction}",
        "matched_video_titles": [str(video.get("title") or "") for video in selected],
        "answer_language": request.answer_language,
        "session_id": session_id,
        "transcript_source": "sqlite_video_metadata",
        "transcript_status": {
            "severity": "success",
            "message": (
                f"Answered from cached YouTube analytics metadata ({noun}). "
                "No transcript retrieval was needed."
            ),
        },
    }


_CONTENT_INTENT_PATTERNS = (
    r"\babout\b",
    r"\bsummari[sz]e\b",
    r"\bsummary\b",
    r"\bdiscuss(?:es|ed|ing|ion)?\b",
    r"\bsay(?:s|ing)?\b",
    r"\bsaid\b",
    r"\btalk(?:s|ed|ing)?\b",
    r"\bmention(?:s|ed|ing)?\b",
    r"\bexplain(?:s|ed|ing)?\b",
    r"\bcover(?:s|ed|ing)?\b",
    r"\bcontent\b",
    r"\btopics?\b",
    r"\bthemes?\b",
    r"\bmain\s+(?:point|idea|topic)\b",
    r"\bwhat\s+happens?\b",
    r"\bteach(?:es|ing)?\b",
    r"\bshow(?:s|ed|ing)?\b",
)


def _requests_video_content(question: str) -> bool:
    """Return True when the user is asking what a selected video *contains*.

    Metadata/identity questions such as "What is the first video?" or
    "Which video is the longest?" should not require Chroma. Content questions
    such as "What is the first video about?" or "Summarize the longest video"
    deliberately continue into transcript RAG.
    """
    text = " ".join(str(question or "").lower().split())
    return any(re.search(pattern, text) for pattern in _CONTENT_INTENT_PATTERNS)


def _format_metadata_date(value) -> str:
    """Format a cached ISO publication date for a concise chatbot answer."""
    raw = str(value or "").strip()
    if not raw:
        return "an unknown date"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.strftime("%B %d, %Y").replace(" 0", " ")
    except ValueError:
        return raw


def _format_metadata_duration(value) -> str:
    """Format a duration in seconds without relying on frontend JavaScript."""
    try:
        total = max(0, int(value or 0))
    except (TypeError, ValueError):
        return "unknown duration"
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _answer_video_identity_query(
    question: str,
    videos: list[dict],
    channel_id: str,
    request: "QuestionRequest",
    session_id: str,
    indexed_video_count: int,
) -> dict | None:
    """Answer simple video identity/stat questions from cached metadata only.

    This intentionally runs before Chroma availability checks. It lets the
    chatbot identify the first/latest/longest/shortest video even when that
    video's transcript has never been indexed. If the wording asks about the
    video's content, the function returns None so normal RAG handles it.
    """
    if not videos or _requests_video_content(question) or _is_comparison_question(question):
        return None

    target_ids, query_scope, matched_titles = _metadata_scope(question, videos, channel_id)
    direct_scopes = {"latest_video", "oldest_video", "longest_video", "shortest_video"}
    if query_scope not in direct_scopes or len(target_ids) != 1:
        return None

    target_id = str(target_ids[0])
    video = next((row for row in videos if str(row.get("video_id") or "") == target_id), None)
    if video is None:
        return None

    title = str(video.get("title") or target_id)
    published = _format_metadata_date(video.get("published_at"))
    duration = _format_metadata_duration(video.get("duration_seconds"))
    if query_scope == "latest_video":
        answer = f'The most recent video is "{title}", published on {published}.'
    elif query_scope == "oldest_video":
        answer = f'The first/oldest video is "{title}", published on {published}.'
    elif query_scope == "longest_video":
        answer = f'The longest video is "{title}" at {duration}, published on {published}.'
    else:
        answer = f'The shortest video is "{title}" at {duration}, published on {published}.'

    requested_language = _analytics_answer_language(request.answer_language, question)
    answer = _translate_metadata_answer(answer, requested_language)

    source = {
        "video_id": target_id,
        "title": title,
        "video_url": f"https://www.youtube.com/watch?v={target_id}",
        "published_at": video.get("published_at"),
        "duration_seconds": video.get("duration_seconds"),
        "language": video.get("default_audio_language") or video.get("default_language"),
        "start_time": None,
        "end_time": None,
    }

    # Preserve the selected video so a follow-up like "What is it about?"
    # can switch naturally from metadata into transcript RAG.
    with _CHAT_LOCK:
        active = _CHAT_SESSIONS.get(session_id)
        if active is not None:
            active["messages"].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            active["messages"] = active["messages"][-_MAX_CHAT_MESSAGES:]
            active["target_video_ids"] = [target_id]
            active["updated_at"] = datetime.now(timezone.utc).isoformat()

    return {
        "answer": answer,
        "sources": [source],
        "retrieved_chunk_count": 0,
        "mode": "video_metadata_identity",
        "indexed_video_count": indexed_video_count,
        "rag_storage": "chroma",
        "rag_test_mode": RAG_TEST_MODE,
        "query_scope": query_scope,
        "matched_video_titles": matched_titles or [title],
        "answer_language": request.answer_language,
        "session_id": session_id,
        "transcript_source": "sqlite_video_metadata",
        "transcript_status": {
            "severity": "success",
            "message": "Answered from cached video metadata. No transcript or Chroma lookup was required.",
        },
    }


# In-memory chat context. This intentionally disappears when FastAPI restarts.
_CHAT_SESSIONS: dict[str, dict] = {}
_CHAT_LOCK = threading.Lock()
_MAX_CHAT_SESSIONS = 100
_MAX_CHAT_MESSAGES = 12

# In-memory background indexing jobs. They are operational state, not project data.
_INDEX_JOBS: dict[str, dict] = {}
_INDEX_JOB_LOCK = threading.Lock()

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_LANGUAGE_CODES = {
    "english": {"en"}, "spanish": {"es"}, "telugu": {"te"},
    "hindi": {"hi"}, "french": {"fr"}, "german": {"de"},
    "portuguese": {"pt"}, "japanese": {"ja"}, "korean": {"ko"},
    "arabic": {"ar"}, "chinese": {"zh"}, "tamil": {"ta"},
    "kannada": {"kn"}, "malayalam": {"ml"}, "bengali": {"bn"},
    "punjabi": {"pa"}, "marathi": {"mr"}, "urdu": {"ur"},
}

_RANK_COUNT_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _ranked_video_list_request(question: str) -> tuple[int, str] | None:
    """Detect ranked metadata-list questions such as 'top 5 most recent videos'.

    These questions should be answered from indexed-video metadata, not from
    transcript text. Returning None leaves normal semantic RAG behavior intact.
    """
    text = str(question or "").strip().lower()
    if not any(token in text for token in ("video", "videos", "upload", "uploads")):
        return None

    rank_mode = None
    if any(token in text for token in ("most recent", "latest", "newest")):
        rank_mode = "latest"
    elif any(token in text for token in ("oldest", "earliest")):
        rank_mode = "oldest"
    elif "longest" in text:
        rank_mode = "longest"
    elif "shortest" in text:
        rank_mode = "shortest"

    if rank_mode is None:
        return None

    count = None
    digit_match = re.search(r"\b(?:top\s+)?(\d{1,5})\b", text)
    if digit_match:
        count = int(digit_match.group(1))
    else:
        for word, value in _RANK_COUNT_WORDS.items():
            if re.search(rf"\b(?:top\s+)?{word}\b", text):
                count = value
                break

    # A singular/latest question continues through the existing one-video route.
    if count is None or count < 2:
        return None

    return count, rank_mode


def _sort_ranked_videos(videos: list[dict], rank_mode: str) -> list[dict]:
    """Sort cached/indexed videos for metadata-list questions."""
    if rank_mode == "latest":
        return sorted(videos, key=lambda v: str(v.get("published_at") or ""), reverse=True)
    if rank_mode == "oldest":
        return sorted(videos, key=lambda v: str(v.get("published_at") or ""))
    if rank_mode == "longest":
        return sorted(videos, key=lambda v: int(v.get("duration_seconds") or 0), reverse=True)
    if rank_mode == "shortest":
        return sorted(videos, key=lambda v: int(v.get("duration_seconds") or 0))
    return list(videos)


def _video_language(video: dict, transcript_languages: dict[str, str]) -> str:
    return str(
        transcript_languages.get(str(video.get("video_id")))
        or video.get("default_audio_language")
        or video.get("default_language")
        or ""
    ).lower()


def _requested_video_language(question: str) -> str | None:
    """Return a language only when the user clearly means video/transcript language.

    A response-language instruction such as ``Answer in Japanese`` must not change
    the video scope.  Language metadata filtering is therefore intentionally
    contextual instead of triggering on any language name anywhere in the
    question.
    """
    text = str(question or "").strip().lower()

    for language in _LANGUAGE_CODES:
        escaped = re.escape(language)
        patterns = (
            # "Spanish video", "latest Spanish videos", "Spanish-language video"
            rf"\b{escaped}(?:[- ]language)?\s+(?:video|videos|upload|uploads)\b",
            # "videos in Spanish" / "uploads in Japanese"
            rf"\b(?:video|videos|upload|uploads)\s+(?:in|spoken in)\s+{escaped}\b",
            # "transcript in Spanish" / "transcripts are in Telugu"
            rf"\b(?:transcript|transcripts|captions|caption)\s+(?:is |are )?(?:in|spoken in)\s+{escaped}\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            return language

    return None


def find_requested_videos(question: str, videos: list[dict]) -> list[dict]:
    """Return all clearly named videos in a question, longest title first."""
    clean_question = str(question or "")
    normalised_question = _normalise_title_text(clean_question)
    matches: dict[str, dict] = {}

    for video in videos:
        video_id = str(video.get("video_id") or "")
        title = str(video.get("title") or "").strip()
        normalised_title = _normalise_title_text(title)
        if len(video_id) >= 8 and video_id in clean_question:
            matches[video_id] = video
        elif len(normalised_title) >= 4 and normalised_title in normalised_question:
            matches[video_id] = video

    for phrase in re.findall(r'["“”]([^"“”]{3,})["“”]', clean_question):
        normalised_phrase = _normalise_title_text(phrase)
        candidates = []
        for video in videos:
            title_norm = _normalise_title_text(video.get("title") or "")
            if normalised_phrase and (normalised_phrase in title_norm or title_norm in normalised_phrase):
                candidates.append(video)
        if len(candidates) == 1:
            matches[str(candidates[0].get("video_id"))] = candidates[0]

    return sorted(matches.values(), key=lambda v: len(str(v.get("title") or "")), reverse=True)


def _is_comparison_question(question: str) -> bool:
    """Return True when the user is explicitly asking to compare video content."""
    text = str(question or "").strip().lower()
    return any(
        token in text
        for token in (
            "compare", "comparison", "versus", " vs ", "difference",
            "differences", "similarity", "similarities", "contrast",
        )
    )


def _metadata_scope(question: str, videos: list[dict], channel_id: str) -> tuple[list[str], str, list[str]]:
    """Resolve metadata-aware video scopes before semantic retrieval."""
    text = str(question or "").lower()
    named = find_requested_videos(question, videos)
    comparison_intent = _is_comparison_question(question)
    if len(named) >= 2 and comparison_intent:
        named_ids = [str(v["video_id"]) for v in named]
        named_titles = [str(v.get("title") or "") for v in named]
        return named_ids, "video_comparison", named_titles

    # "Compare all/every indexed videos" should use the whole indexed scope,
    # not only the globally nearest transcript chunks.
    if comparison_intent and any(token in text for token in ("all videos", "all indexed", "every video", "entire channel")):
        indexed_ids = get_indexed_video_ids(channel_id)
        selected = [v for v in videos if str(v.get("video_id") or "") in indexed_ids]
        return (
            [str(v["video_id"]) for v in selected],
            "all_indexed_video_comparison",
            [str(v.get("title") or "") for v in selected],
        )

    if len(named) == 1:
        return [str(named[0]["video_id"])], "specific_video", [str(named[0].get("title") or "")]

    # Likes/views/comments are analytics metadata, but they can also establish
    # a RAG scope for content questions. Example: "What does the most liked
    # video discuss?" selects the most-liked video here, then transcript RAG
    # answers what that video actually says.
    analytics_rank = _analytics_rank_request(question)
    if analytics_rank and _requests_video_content(question):
        requested_count, metric, direction = analytics_rank
        selected = sorted(
            videos,
            key=lambda video: int(video.get(metric) or 0),
            reverse=(direction == "desc"),
        )[:requested_count]
        if selected:
            suffix = "comparison" if len(selected) > 1 else "video"
            return (
                [str(video["video_id"]) for video in selected],
                f"analytics_{metric}_{direction}_{suffix}",
                [str(video.get("title") or "") for video in selected],
            )

    transcript_languages = {
        str(row.get("video_id")): str(row.get("language") or "")
        for row in get_saved_transcripts_for_channel(channel_id, include_text=False)
    }
    candidates = list(videos)

    # Optional video/transcript-language constraint.  Do not treat response-language
    # instructions such as "Answer in Japanese" as a video filter.
    requested_language = _requested_video_language(question)
    if requested_language:
        codes = _LANGUAGE_CODES[requested_language]
        filtered = []
        for video in candidates:
            language = _video_language(video, transcript_languages)
            base = language.split("-")[0]
            if language in codes or base in codes or requested_language in language:
                filtered.append(video)
        if filtered:
            candidates = filtered

    # Optional month/year constraint.
    requested_month = next((number for name, number in _MONTHS.items() if name in text), None)
    year_match = re.search(r"\b(20\d{2})\b", text)
    requested_year = int(year_match.group(1)) if year_match else None
    if requested_month or requested_year:
        dated = []
        for video in candidates:
            try:
                value = str(video.get("published_at") or "").replace("Z", "+00:00")
                published = datetime.fromisoformat(value)
            except (TypeError, ValueError):
                continue
            if requested_month and published.month != requested_month:
                continue
            if requested_year and published.year != requested_year:
                continue
            dated.append(video)
        if dated:
            candidates = dated
            scope = "date_filtered_video_comparison" if comparison_intent else "date_filtered_videos"
            return [str(v["video_id"]) for v in candidates], scope, [str(v.get("title") or "") for v in candidates]

    dated = [v for v in candidates if v.get("published_at")]
    if ("video" in text or "upload" in text) and any(x in text for x in ("most recent", "latest", "newest")):
        target = max(dated, key=lambda v: str(v.get("published_at"))) if dated else (candidates[0] if candidates else None)
        if target:
            return [str(target["video_id"])], "latest_video", [str(target.get("title") or "")]
    if ("video" in text or "upload" in text) and any(x in text for x in ("oldest", "earliest", "first video")):
        target = min(dated, key=lambda v: str(v.get("published_at"))) if dated else (candidates[-1] if candidates else None)
        if target:
            return [str(target["video_id"])], "oldest_video", [str(target.get("title") or "")]
    if "video" in text and "longest" in text and candidates:
        target = max(candidates, key=lambda v: int(v.get("duration_seconds") or 0))
        return [str(target["video_id"])], "longest_video", [str(target.get("title") or "")]
    if "video" in text and "shortest" in text and candidates:
        target = min(candidates, key=lambda v: int(v.get("duration_seconds") or 0))
        return [str(target["video_id"])], "shortest_video", [str(target.get("title") or "")]

    if requested_language and candidates:
        scope = "language_filtered_video_comparison" if comparison_intent else "language_filtered_videos"
        return [str(v["video_id"]) for v in candidates], scope, [str(v.get("title") or "") for v in candidates]
    return [], "channel", []


def _looks_like_follow_up(question: str) -> bool:
    text = str(question or "").strip().lower()
    prefixes = ("what about", "how about", "and what", "and how", "why", "how so", "does it", "do they", "what does he", "what does she", "what does it", "can you explain that", "tell me more")
    return text.startswith(prefixes) or bool(re.search(r"\b(it|that|this|they|he|she|those|these|second one|first one)\b", text))


def _get_chat_session(session_id: str | None, channel_id: str) -> tuple[str, dict]:
    sid = str(session_id or "").strip() or str(uuid.uuid4())
    with _CHAT_LOCK:
        if sid not in _CHAT_SESSIONS and len(_CHAT_SESSIONS) >= _MAX_CHAT_SESSIONS:
            oldest = min(_CHAT_SESSIONS, key=lambda key: _CHAT_SESSIONS[key].get("updated_at", ""))
            _CHAT_SESSIONS.pop(oldest, None)
        session = _CHAT_SESSIONS.setdefault(sid, {"channel_id": channel_id, "messages": [], "target_video_ids": [], "updated_at": ""})
        if session.get("channel_id") != channel_id:
            session.clear()
            session.update({"channel_id": channel_id, "messages": [], "target_video_ids": [], "updated_at": ""})
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        return sid, session


def _job_update(job_id: str, **changes) -> None:
    with _INDEX_JOB_LOCK:
        job = _INDEX_JOBS.get(job_id)
        if job is not None:
            job.update(changes)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()


def _job_cancelled(job_id: str) -> bool:
    with _INDEX_JOB_LOCK:
        return bool(_INDEX_JOBS.get(job_id, {}).get("cancel_requested"))


def _run_index_job(job_id: str, channel_id: str, requested_video_id: str | None) -> None:
    """Background RAG build with incremental transcript/index status."""
    try:
        _job_update(job_id, state="running", phase="loading_channel", message="Loading channel metadata...")
        result = get_channel_data(channel_id)
        if result is None:
            raise RuntimeError("YouTube channel was not found.")

        videos = select_rag_videos(result["videos"])
        if requested_video_id:
            videos = [v for v in result["videos"] if str(v.get("video_id")) == requested_video_id]
            if not videos:
                raise RuntimeError("Requested video was not found in this channel.")

        if channel_requires_embedding_rebuild(channel_id):
            _job_update(job_id, phase="rebuilding", message="Rebuilding Chroma for the current embedding model...")
            all_videos = result["videos"]
            previous_ids = get_indexed_video_ids(channel_id, [v["video_id"] for v in all_videos])
            selected_ids = {str(v["video_id"]) for v in videos}
            rebuild_ids = previous_ids | selected_ids
            rebuild_videos = [v for v in all_videos if str(v.get("video_id")) in rebuild_ids]
            cached = get_cached_video_transcripts(rebuild_videos)
            missing_old = set(cached["missing_video_ids"]) & previous_ids
            if missing_old:
                raise RuntimeError("Cannot safely rebuild because previously indexed transcripts are missing from SQLite: " + ", ".join(sorted(missing_old)))
            delete_channel_index(channel_id)
            videos = rebuild_videos

        enrich_indexed_video_metadata(channel_id, videos, channel_title=result["channel"].get("title"))
        missing_videos, indexed_before = get_missing_rag_videos(channel_id, videos)
        _job_update(job_id, total_video_count=len(videos), already_indexed_count=len(indexed_before), remaining_count=len(missing_videos), phase="transcripts")

        counters = {"processed": 0, "newly_indexed": 0, "chunks": 0, "cached": 0, "youtube": 0, "failed": 0}

        def on_progress(event):
            _job_update(job_id, current_video_id=event.get("video_id"), current_title=event.get("title"), current_position=event.get("position"), message=f"{event.get('event', 'processing').replace('_', ' ').title()}: {event.get('title') or event.get('video_id')}")

        def on_transcript(record, info):
            if _job_cancelled(job_id):
                return
            _job_update(job_id, phase="embedding", current_video_id=record.get("video_id"), current_title=record.get("title"), message=f"Embedding: {record.get('title')}")
            summary = index_transcripts(channel_id, [record], channel_title=result["channel"].get("title"))
            counters["processed"] += 1
            counters["newly_indexed"] += int(summary.get("indexed_video_count", 0))
            counters["chunks"] += int(summary.get("indexed_chunk_count", 0))
            counters["cached"] += 1 if info.get("source") == "sqlite_cache" else 0
            counters["youtube"] += 1 if info.get("source") == "youtube_transcript_api" else 0
            _job_update(job_id, phase="transcripts", processed_count=counters["processed"], newly_indexed_count=counters["newly_indexed"], indexed_chunk_count=counters["chunks"], sqlite_cache_count=counters["cached"], youtube_fetch_count=counters["youtube"], remaining_count=max(0, len(missing_videos)-counters["processed"]))

        if missing_videos:
            batch = get_video_transcripts_with_status(missing_videos, channel_id, should_stop=lambda: _job_cancelled(job_id), on_transcript=on_transcript, on_progress=on_progress)
            counters["failed"] = int(batch.get("failed_video_count", 0))
        else:
            batch = {"blocked": False, "cancelled": False, "failed_video_count": 0}

        indexed_after = get_indexed_video_ids(channel_id, [v["video_id"] for v in videos])
        if batch.get("blocked"):
            final_state, severity = "blocked", "warning"
            message = "YouTube blocked additional transcript requests. Completed work was preserved."
        elif batch.get("cancelled") or _job_cancelled(job_id):
            final_state, severity = "cancelled", "info"
            message = "Indexing stopped. Completed transcripts and Chroma chunks were preserved; run indexing again to resume missing videos."
        else:
            final_state, severity = "completed", "success"
            message = f"Indexing complete: {len(indexed_after)} of {len(videos)} selected videos are indexed."
        _job_update(job_id, state=final_state, phase="done", severity=severity, message=message, indexed_video_count=len(indexed_after), failed_video_count=counters["failed"], remaining_count=max(0, len(videos)-len(indexed_after)), finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        _job_update(job_id, state="failed", phase="done", severity="warning", message=f"Indexing failed: {error}", error=str(error), finished_at=datetime.now(timezone.utc).isoformat())


@app.post("/api/rag/index/start")
def start_rag_index_job(request: RagIndexRequest):
    channel_id = request.channel_id.strip()
    if not channel_id:
        raise HTTPException(status_code=400, detail="Channel ID is required.")
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "channel_id": channel_id, "requested_video_id": request.video_id, "state": "queued", "phase": "queued", "severity": "info", "message": "Indexing queued...", "cancel_requested": False, "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "processed_count": 0, "indexed_video_count": 0, "remaining_count": 0}
    with _INDEX_JOB_LOCK:
        if any(existing.get("channel_id") == channel_id and existing.get("state") in {"queued", "running"} for existing in _INDEX_JOBS.values()):
            raise HTTPException(status_code=409, detail="An indexing job is already running for this channel.")
        _INDEX_JOBS[job_id] = job
    threading.Thread(target=_run_index_job, args=(job_id, channel_id, request.video_id), daemon=True).start()
    return job


@app.get("/api/rag/index/job/{job_id}")
def rag_index_job_status(job_id: str):
    with _INDEX_JOB_LOCK:
        job = _INDEX_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Indexing job was not found.")
        return dict(job)


@app.post("/api/rag/index/job/{job_id}/cancel")
def cancel_rag_index_job(job_id: str):
    with _INDEX_JOB_LOCK:
        job = _INDEX_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Indexing job was not found.")
        job["cancel_requested"] = True
        job["message"] = "Stop requested. Finishing the current safe step..."
        return dict(job)


@app.get("/api/channel/{channel_id}/video-statuses")
def video_pipeline_statuses(channel_id: str):
    channel = get_cached_channel_data(channel_id)
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel is not cached yet.")
    video_ids = [str(v.get("video_id")) for v in channel["videos"] if v.get("video_id")]
    cached_ids = {str(row.get("video_id")) for row in get_saved_transcripts_for_channel(channel_id, include_text=False)}
    indexed_ids = get_indexed_video_ids(channel_id, video_ids)
    processing: dict[str, str] = {}
    with _INDEX_JOB_LOCK:
        for job in _INDEX_JOBS.values():
            if job.get("channel_id") == channel_id and job.get("state") in {"queued", "running"} and job.get("current_video_id"):
                processing[str(job["current_video_id"])] = str(job.get("phase") or "processing")
    return {"channel_id": channel_id, "videos": [{"video_id": vid, "transcript_cached": vid in cached_ids, "rag_indexed": vid in indexed_ids, "processing_phase": processing.get(vid)} for vid in video_ids]}


@app.delete("/api/chat/session/{session_id}")
def clear_chat_session(session_id: str):
    with _CHAT_LOCK:
        _CHAT_SESSIONS.pop(session_id, None)
    return {"status": "success"}


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
    if request.video_id:
        requested_video_id = request.video_id.strip()
        videos = [video for video in result["videos"] if str(video.get("video_id")) == requested_video_id]
        if not videos:
            raise HTTPException(status_code=404, detail="Requested video was not found in this channel.")

    if RAG_TEST_MODE:
        rag_channel_id = f"test::{channel_id}"
        test_transcripts = load_test_transcripts()
        for item in test_transcripts:
            if isinstance(item, dict):
                item.setdefault("transcript_source", "test_file")

        try:
            if channel_requires_embedding_rebuild(rag_channel_id):
                delete_channel_index(rag_channel_id)
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

    rebuilt_for_embedding_model = False
    try:
        # Multilingual retrieval requires transcript chunks and questions to be
        # embedded by the same model. If an older index exists, preserve every
        # previously indexed video during the rebuild even when the current
        # RAG_TRANSCRIPT_VIDEO_LIMIT is smaller than the old index.
        if channel_requires_embedding_rebuild(channel_id):
            all_videos = result["videos"]
            previous_indexed_ids = get_indexed_video_ids(
                channel_id,
                [video["video_id"] for video in all_videos],
            )
            selected_ids = {str(video["video_id"]) for video in videos}
            rebuild_ids = previous_indexed_ids | selected_ids
            rebuild_videos = [
                video for video in all_videos
                if str(video.get("video_id")) in rebuild_ids
            ]

            # A normally indexed video should always have its raw transcript in
            # SQLite. Refuse to destroy the old Chroma copy if that invariant is
            # broken, because re-downloading it may be blocked by YouTube.
            cached_check = get_cached_video_transcripts(rebuild_videos)
            missing_cached = set(cached_check["missing_video_ids"]) & previous_indexed_ids
            if missing_cached:
                raise RuntimeError(
                    "Cannot safely rebuild the multilingual RAG index because "
                    "some previously indexed transcripts are missing from SQLite: "
                    + ", ".join(sorted(missing_cached))
                )

            delete_channel_index(channel_id)
            videos = rebuild_videos
            rebuilt_for_embedding_model = True

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
        "embedding_model": EMBEDDING_MODEL_NAME,
        "rebuilt_for_embedding_model": rebuilt_for_embedding_model,
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
    """Answer from existing Chroma only, with metadata routing and chat context."""
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

    cached_channel = get_cached_channel_data(channel_id) if not RAG_TEST_MODE else None
    cached_videos = (cached_channel or {}).get("videos", [])
    session_id, session = _get_chat_session(request.session_id, channel_id)
    history = list(session.get("messages", []))

    # Analytics facts such as likes/views/comments live in SQLite video metadata,
    # not in transcript chunks. Answer these locally before requiring a RAG index.
    if not RAG_TEST_MODE:
        analytics_result = _answer_analytics_rank_query(
            question=question,
            videos=cached_videos,
            request=request,
            session_id=session_id,
            indexed_video_count=indexed_video_count,
        )
        if analytics_result is not None:
            return analytics_result

        identity_result = _answer_video_identity_query(
            question=question,
            videos=cached_videos,
            channel_id=channel_id,
            request=request,
            session_id=session_id,
            indexed_video_count=indexed_video_count,
        )
        if identity_result is not None:
            return identity_result

    if int(stats.get("chunk_count", 0)) == 0:
        return {"answer": "No transcript content has been indexed for this channel yet. Use 'Index RAG Content' first.", "sources": [], "retrieved_chunk_count": 0, "mode": "no_indexed_content", "indexed_video_count": 0, "rag_storage": "chroma", "rag_test_mode": RAG_TEST_MODE, "query_scope": "channel", "session_id": session_id, "transcript_status": {"severity": "info", "message": "No RAG transcript content is indexed yet."}}
    if channel_requires_embedding_rebuild(rag_channel_id):
        return {"answer": "This channel's RAG index uses an older embedding model. Index RAG Content once to rebuild it from cached transcripts.", "sources": [], "retrieved_chunk_count": 0, "mode": "embedding_rebuild_required", "indexed_video_count": indexed_video_count, "rag_storage": "chroma", "rag_test_mode": RAG_TEST_MODE, "query_scope": "channel", "session_id": session_id, "transcript_status": {"severity": "info", "message": "Rebuild the RAG index for the current embedding model."}}

    target_video_ids: list[str] = []
    query_scope = "channel"
    matched_titles: list[str] = []

    # Ranked list questions such as "top 5 most recent video titles" are
    # metadata questions. Transcript chunks normally do not contain upload
    # order or the complete video-title list, so answer these directly from
    # the cached YouTube metadata limited to videos that are actually indexed
    # in Chroma. This keeps the answer aligned with the current RAG database.
    comparison_intent = _is_comparison_question(question)
    ranked_request = None if RAG_TEST_MODE else _ranked_video_list_request(question)
    ranked_comparison_ids: list[str] = []
    ranked_comparison_titles: list[str] = []
    ranked_comparison_scope = ""
    if ranked_request and cached_videos:
        requested_count, rank_mode = ranked_request
        indexed_ids = get_indexed_video_ids(rag_channel_id)
        indexed_videos = [
            video for video in cached_videos
            if str(video.get("video_id") or "") in indexed_ids
        ]
        ranked_videos = _sort_ranked_videos(indexed_videos, rank_mode)
        selected = ranked_videos[:requested_count]

        if not selected:
            return {
                "answer": "No indexed videos are available for that metadata query yet.",
                "sources": [],
                "retrieved_chunk_count": 0,
                "mode": "metadata_list",
                "indexed_video_count": indexed_video_count,
                "rag_storage": "chroma",
                "rag_test_mode": RAG_TEST_MODE,
                "query_scope": f"{rank_mode}_video_list",
                "session_id": session_id,
                "transcript_status": {
                    "severity": "info",
                    "message": "The question was recognized as a video-metadata list query."
                },
            }

        if comparison_intent:
            ranked_comparison_ids = [str(video.get("video_id") or "") for video in selected]
            ranked_comparison_titles = [str(video.get("title") or "") for video in selected]
            ranked_comparison_scope = f"{rank_mode}_video_comparison"
        else:
            labels = {
                "latest": "most recent",
                "oldest": "oldest",
                "longest": "longest",
                "shortest": "shortest",
            }
            label = labels.get(rank_mode, rank_mode)
            lines = [
                f"{index}. {str(video.get('title') or video.get('video_id') or 'Untitled video')}"
                for index, video in enumerate(selected, start=1)
            ]
            answer = f"Top {len(selected)} {label} indexed video titles:\n" + "\n".join(lines)
            if len(selected) < requested_count:
                answer += (
                    f"\n\nOnly {len(selected)} indexed video(s) are currently available "
                    f"out of the {requested_count} requested."
                )
    
            sources = [
                {
                    "video_id": str(video.get("video_id") or ""),
                    "title": str(video.get("title") or video.get("video_id") or "Untitled video"),
                    "video_url": f"https://www.youtube.com/watch?v={str(video.get('video_id') or '')}",
                    "published_at": video.get("published_at"),
                    "duration_seconds": video.get("duration_seconds"),
                    "language": video.get("default_audio_language") or video.get("default_language"),
                    "start_time": None,
                    "end_time": None,
                }
                for video in selected
            ]
            selected_ids = [str(video.get("video_id") or "") for video in selected]
    
            # Preserve the ordered list in the in-memory chat session so a follow-up
            # like "What is the second one about?" can resolve to the correct video.
            with _CHAT_LOCK:
                active = _CHAT_SESSIONS.get(session_id)
                if active is not None:
                    active["messages"].extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ])
                    active["messages"] = active["messages"][-_MAX_CHAT_MESSAGES:]
                    active["target_video_ids"] = selected_ids
                    active["updated_at"] = datetime.now(timezone.utc).isoformat()
    
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunk_count": 0,
                "mode": "metadata_list",
                "indexed_video_count": indexed_video_count,
                "rag_storage": "chroma",
                "rag_test_mode": RAG_TEST_MODE,
                "query_scope": f"{rank_mode}_video_list",
                "matched_video_titles": [str(video.get("title") or "") for video in selected],
                "answer_language": request.answer_language,
                "session_id": session_id,
                "transcript_source": "indexed_video_metadata",
                "transcript_status": {
                    "severity": "success",
                    "message": (
                        f"Answered from metadata for {len(selected)} indexed video(s). "
                        "No transcript retrieval was needed."
                    ),
                },
            }
    if request.video_id:
        target_video_ids = [request.video_id.strip()]
        query_scope = "selected_video"
    elif ranked_comparison_ids:
        target_video_ids = ranked_comparison_ids
        query_scope = ranked_comparison_scope
        matched_titles = ranked_comparison_titles
    elif not RAG_TEST_MODE:
        target_video_ids, query_scope, matched_titles = _metadata_scope(question, cached_videos, channel_id)

    # Conversational follow-ups inherit the previous video scope only when the
    # new question does not explicitly establish another scope.
    if not target_video_ids and _looks_like_follow_up(question) and session.get("target_video_ids"):
        previous_targets = list(session["target_video_ids"])
        lower_question = question.lower()
        ordinal_map = {"first": 0, "second": 1, "third": 2}
        selected_index = next((index for word, index in ordinal_map.items() if word in lower_question), None)
        if selected_index is not None and selected_index < len(previous_targets):
            target_video_ids = [previous_targets[selected_index]]
        else:
            target_video_ids = previous_targets
        query_scope = "conversation_follow_up"

    if target_video_ids and not RAG_TEST_MODE:
        indexed = get_indexed_video_ids(channel_id, target_video_ids)
        missing = [vid for vid in target_video_ids if vid not in indexed]
        if missing:
            names = [str(v.get("title") or v.get("video_id")) for v in cached_videos if str(v.get("video_id")) in missing]
            return {"answer": "The requested video scope is not fully indexed for RAG yet: " + ", ".join(names or missing), "sources": [], "retrieved_chunk_count": 0, "mode": "target_video_not_indexed", "indexed_video_count": indexed_video_count, "rag_storage": "chroma", "rag_test_mode": RAG_TEST_MODE, "query_scope": query_scope, "matched_video_titles": matched_titles, "session_id": session_id, "transcript_status": {"severity": "info", "message": "Index the requested video(s) first. No YouTube request was made by Q&A."}}

    try:
        rag_result = run_rag_question(
            channel_id=rag_channel_id,
            question=question,
            transcripts=[],
            target_video_ids=target_video_ids or None,
            answer_language=request.answer_language,
            conversation_history=history,
            comparison_mode=(comparison_intent or "comparison" in query_scope),
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error))
    rag_result.pop("rag_prompt", None)

    source_ids = [str(source.get("video_id")) for source in rag_result.get("sources", []) if source.get("video_id")]
    with _CHAT_LOCK:
        active = _CHAT_SESSIONS.get(session_id)
        if active is not None:
            active["messages"].extend([{"role": "user", "content": question}, {"role": "assistant", "content": str(rag_result.get("answer") or "")}])
            active["messages"] = active["messages"][-_MAX_CHAT_MESSAGES:]
            if target_video_ids:
                active["target_video_ids"] = list(target_video_ids)
            elif source_ids:
                active["target_video_ids"] = list(dict.fromkeys(source_ids))
            active["updated_at"] = datetime.now(timezone.utc).isoformat()

    return {**rag_result, "indexed_video_count": indexed_video_count, "transcript_source": "chroma_only", "transcript_status": {"severity": "success", "message": f"RAG searched existing Chroma content across {len(target_video_ids) if target_video_ids else indexed_video_count} video scope item(s). No transcript request was made."}, "rag_storage": "chroma", "rag_test_mode": RAG_TEST_MODE, "query_scope": query_scope, "matched_video_titles": matched_titles, "answer_language": request.answer_language, "session_id": session_id}


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


@app.get("/api/channel/{channel_id}/cached-transcripts")
def cached_transcript_list(channel_id: str):
    """List transcripts already stored in SQLite without contacting YouTube."""
    rows = get_saved_transcripts_for_channel(channel_id, include_text=False)
    items = []
    for row in rows:
        segments = []
        if row.get("segments_json"):
            try:
                decoded = json.loads(row["segments_json"])
                if isinstance(decoded, list):
                    segments = decoded
            except (TypeError, json.JSONDecodeError):
                segments = []

        items.append({
            "video_id": row["video_id"],
            "title": row.get("title") or "Unknown video",
            "language": row.get("language") or row.get("default_audio_language") or row.get("default_language"),
            "published_at": row.get("published_at"),
            "duration_seconds": row.get("duration_seconds"),
            "has_timestamps": bool(segments),
            "transcript_last_updated": row.get("transcript_last_updated"),
        })

    return {"channel_id": channel_id, "transcript_count": len(items), "transcripts": items}


@app.get("/api/transcript/{video_id}")
def cached_transcript(video_id: str):
    """Return one cached transcript for the viewer; never fetch from YouTube."""
    row = get_saved_transcript_with_video(video_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Cached transcript was not found.")

    segments = []
    if row.get("segments_json"):
        try:
            decoded = json.loads(row["segments_json"])
            if isinstance(decoded, list):
                segments = decoded
        except (TypeError, json.JSONDecodeError):
            segments = []

    return {
        "video_id": row["video_id"],
        "channel_id": row["channel_id"],
        "title": row.get("title") or "Unknown video",
        "language": row.get("language") or row.get("default_audio_language") or row.get("default_language"),
        "published_at": row.get("published_at"),
        "duration_seconds": row.get("duration_seconds"),
        "transcript": row.get("transcript") or "",
        "segments": segments,
        "has_timestamps": bool(segments),
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
