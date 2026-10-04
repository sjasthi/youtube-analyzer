"""SQLite persistence and cache helpers for the YouTube Channel Analyzer.

This module stores channel and video data locally so the application does not
need to download the same full channel from YouTube every time it is analyzed.

SQLite is part of Python's standard library, so no extra package is required.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Project root: youtube-channel-analyzer/
BASE_DIR = Path(__file__).resolve().parent.parent

# Keep generated application data in its own folder.
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "youtube_analyzer.db"

# If a saved channel was updated less than this many minutes ago, use the
# database instead of downloading the entire channel again.
CACHE_MAX_AGE_MINUTES = 60


def get_connection():
    """Open a SQLite connection and return rows that behave like dictionaries."""

    # Create data/ the first time the application runs.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    # SQLite does not enforce foreign keys unless this is enabled per connection.
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database():
    """Create the database tables and indexes if they do not already exist."""

    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                custom_url TEXT,
                published_at TEXT,
                thumbnail_url TEXT,
                subscriber_count INTEGER NOT NULL DEFAULT 0,
                video_count INTEGER NOT NULL DEFAULT 0,
                view_count INTEGER NOT NULL DEFAULT 0,
                uploads_playlist_id TEXT,
                last_updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                title TEXT,
                description TEXT,
                published_at TEXT,
                thumbnail_url TEXT,
                duration TEXT,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                view_count INTEGER NOT NULL DEFAULT 0,
                like_count INTEGER NOT NULL DEFAULT 0,
                comment_count INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL,
                FOREIGN KEY (channel_id)
                    REFERENCES channels(channel_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_videos_channel_id
                ON videos(channel_id);

            CREATE INDEX IF NOT EXISTS idx_videos_published_at
                ON videos(published_at);
            """
        )


def save_channel_and_videos(channel, videos):
    """Insert or update one channel and all of its currently public videos.

    SQLite's ON CONFLICT ... DO UPDATE works like an "upsert":
    - new channel/video -> INSERT
    - existing channel/video -> UPDATE

    Because get_all_videos() performs a full channel sync, videos that were
    previously stored but are no longer returned by YouTube are removed from
    this cache.
    """

    now = datetime.now(timezone.utc).isoformat()
    channel_id = channel["channel_id"]

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO channels (
                channel_id,
                title,
                description,
                custom_url,
                published_at,
                thumbnail_url,
                subscriber_count,
                video_count,
                view_count,
                uploads_playlist_id,
                last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                custom_url = excluded.custom_url,
                published_at = excluded.published_at,
                thumbnail_url = excluded.thumbnail_url,
                subscriber_count = excluded.subscriber_count,
                video_count = excluded.video_count,
                view_count = excluded.view_count,
                uploads_playlist_id = excluded.uploads_playlist_id,
                last_updated = excluded.last_updated
            """,
            (
                channel_id,
                channel.get("title"),
                channel.get("description"),
                channel.get("custom_url"),
                channel.get("published_at"),
                channel.get("thumbnail_url"),
                channel.get("subscriber_count", 0),
                channel.get("video_count", 0),
                channel.get("view_count", 0),
                channel.get("uploads_playlist_id"),
                now,
            ),
        )

        current_video_ids = []

        for video in videos:
            video_id = video["video_id"]
            current_video_ids.append(video_id)

            connection.execute(
                """
                INSERT INTO videos (
                    video_id,
                    channel_id,
                    title,
                    description,
                    published_at,
                    thumbnail_url,
                    duration,
                    duration_seconds,
                    view_count,
                    like_count,
                    comment_count,
                    last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    channel_id = excluded.channel_id,
                    title = excluded.title,
                    description = excluded.description,
                    published_at = excluded.published_at,
                    thumbnail_url = excluded.thumbnail_url,
                    duration = excluded.duration,
                    duration_seconds = excluded.duration_seconds,
                    view_count = excluded.view_count,
                    like_count = excluded.like_count,
                    comment_count = excluded.comment_count,
                    last_updated = excluded.last_updated
                """,
                (
                    video_id,
                    channel_id,
                    video.get("title"),
                    video.get("description"),
                    video.get("published_at"),
                    video.get("thumbnail_url"),
                    video.get("duration"),
                    video.get("duration_seconds", 0),
                    video.get("view_count", 0),
                    video.get("like_count", 0),
                    video.get("comment_count", 0),
                    now,
                ),
            )

        # This database currently acts as a cache of the channel's current
        # public uploads. Remove cached rows no longer present in a full sync.
        if current_video_ids:
            placeholders = ",".join("?" for _ in current_video_ids)
            connection.execute(
                f"""
                DELETE FROM videos
                WHERE channel_id = ?
                  AND video_id NOT IN ({placeholders})
                """,
                [channel_id, *current_video_ids],
            )
        else:
            connection.execute(
                "DELETE FROM videos WHERE channel_id = ?",
                (channel_id,),
            )


def get_cached_channel_data(channel_id):
    """Return saved channel/videos using the same structure as get_all_videos()."""

    with get_connection() as connection:
        channel_row = connection.execute(
            "SELECT * FROM channels WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()

        if channel_row is None:
            return None

        video_rows = connection.execute(
            """
            SELECT *
            FROM videos
            WHERE channel_id = ?
            ORDER BY published_at DESC
            """,
            (channel_id,),
        ).fetchall()

    channel = {
        "channel_id": channel_row["channel_id"],
        "title": channel_row["title"],
        "description": channel_row["description"],
        "custom_url": channel_row["custom_url"],
        "published_at": channel_row["published_at"],
        "thumbnail_url": channel_row["thumbnail_url"],
        "subscriber_count": channel_row["subscriber_count"],
        "video_count": channel_row["video_count"],
        "view_count": channel_row["view_count"],
        "uploads_playlist_id": channel_row["uploads_playlist_id"],
    }

    videos = [
        {
            "video_id": row["video_id"],
            "title": row["title"],
            "description": row["description"],
            "published_at": row["published_at"],
            "thumbnail_url": row["thumbnail_url"],
            "duration": row["duration"],
            "duration_seconds": row["duration_seconds"],
            "view_count": row["view_count"],
            "like_count": row["like_count"],
            "comment_count": row["comment_count"],
        }
        for row in video_rows
    ]

    return {
        "channel": channel,
        "videos": videos,
        "retrieved_video_count": len(videos),
        "last_updated": channel_row["last_updated"],
    }


def is_cache_fresh(channel_id, max_age_minutes=CACHE_MAX_AGE_MINUTES):
    """Return True when the channel exists and was refreshed recently enough."""

    with get_connection() as connection:
        row = connection.execute(
            "SELECT last_updated FROM channels WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()

    if row is None:
        return False

    try:
        last_updated = datetime.fromisoformat(row["last_updated"])
    except (TypeError, ValueError):
        return False

    # Old databases may contain a timestamp without timezone information.
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    return last_updated >= cutoff


def get_database_stats():
    """Return small counts that are useful for testing/debugging the database."""

    with get_connection() as connection:
        channel_count = connection.execute(
            "SELECT COUNT(*) AS count FROM channels"
        ).fetchone()["count"]

        video_count = connection.execute(
            "SELECT COUNT(*) AS count FROM videos"
        ).fetchone()["count"]

    return {
        "channel_count": channel_count,
        "video_count": video_count,
        "database_path": str(DATABASE_PATH),
    }
