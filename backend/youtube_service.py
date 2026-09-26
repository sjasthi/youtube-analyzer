"""Functions that communicate with the YouTube Data API."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build

# Locate the project root and load the .env file from there.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_youtube_client():
    """Create the YouTube API client using our API key."""

    # Read the secret API key from the environment instead of hard-coding it.
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY is missing. Add it to the .env file in the project root."
        )

    # This object gives us access to channels(), playlistItems(), videos(), etc.
    return build("youtube", "v3", developerKey=api_key)


def parse_duration_to_seconds(duration):
    """Convert a YouTube duration such as PT12M30S into total seconds."""

    # YouTube uses ISO 8601 duration strings.
    pattern = re.compile(
        r"^PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$"
    )
    match = pattern.match(duration or "")

    if not match:
        return 0

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)

    return hours * 3600 + minutes * 60 + seconds


def get_channel_by_id(channel_id):
    """Retrieve public channel information from YouTube."""

    youtube = get_youtube_client()

    # snippet = title/description/thumbnail
    # statistics = subscribers/views/video count
    # contentDetails = includes the channel's uploads playlist
    response = (
        youtube.channels()
        .list(part="snippet,statistics,contentDetails", id=channel_id)
        .execute()
    )

    # An empty items list means YouTube did not find the requested channel.
    if not response.get("items"):
        return None

    channel = response["items"][0]
    snippet = channel["snippet"]
    statistics = channel["statistics"]
    content_details = channel["contentDetails"]

    # Return only the values our application currently needs.
    return {
        "channel_id": channel["id"],
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "custom_url": snippet.get("customUrl"),
        "published_at": snippet.get("publishedAt"),
        "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url"),
        "subscriber_count": int(statistics.get("subscriberCount", 0)),
        "video_count": int(statistics.get("videoCount", 0)),
        "view_count": int(statistics.get("viewCount", 0)),
        "uploads_playlist_id": content_details.get("relatedPlaylists", {}).get("uploads"),
    }


def get_recent_videos(channel_id, max_results=50):
    """Retrieve up to 50 recent uploaded videos and their statistics."""

    # The YouTube playlistItems endpoint supports at most 50 items per request.
    if max_results < 1 or max_results > 50:
        raise ValueError("max_results must be between 1 and 50.")

    # First get the channel so we can find its uploads playlist ID.
    channel = get_channel_by_id(channel_id)
    if channel is None:
        return None

    uploads_playlist_id = channel.get("uploads_playlist_id")
    if not uploads_playlist_id:
        return {"channel": channel, "videos": [], "retrieved_video_count": 0}

    youtube = get_youtube_client()

    # Step 1: get recent video IDs from the channel's uploads playlist.
    playlist_response = (
        youtube.playlistItems()
        .list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=max_results,
        )
        .execute()
    )

    video_ids = [
        item["contentDetails"]["videoId"]
        for item in playlist_response.get("items", [])
        if item.get("contentDetails", {}).get("videoId")
    ]

    if not video_ids:
        return {"channel": channel, "videos": [], "retrieved_video_count": 0}

    # Step 2: retrieve details/statistics for all of those video IDs.
    videos_response = (
        youtube.videos()
        .list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids),
        )
        .execute()
    )

    # Store results by ID so we can restore playlist order afterward.
    videos_by_id = {}

    for video in videos_response.get("items", []):
        snippet = video.get("snippet", {})
        statistics = video.get("statistics", {})
        content_details = video.get("contentDetails", {})
        duration_iso = content_details.get("duration", "PT0S")

        videos_by_id[video["id"]] = {
            "video_id": video["id"],
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "published_at": snippet.get("publishedAt"),
            "thumbnail_url": snippet.get("thumbnails", {}).get("medium", {}).get("url"),
            "duration": duration_iso,
            "duration_seconds": parse_duration_to_seconds(duration_iso),
            "view_count": int(statistics.get("viewCount", 0)),
            "like_count": int(statistics.get("likeCount", 0)),
            "comment_count": int(statistics.get("commentCount", 0)),
        }

    # Keep the same newest-to-oldest order returned by the uploads playlist.
    ordered_videos = [
        videos_by_id[video_id]
        for video_id in video_ids
        if video_id in videos_by_id
    ]

    return {
        "channel": channel,
        "videos": ordered_videos,
        "retrieved_video_count": len(ordered_videos),
    }
