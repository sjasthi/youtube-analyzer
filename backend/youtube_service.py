"""Functions that communicate with the YouTube Data API."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build

# Locate the project root and load the .env file from there.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# YouTube list requests are handled in groups of up to 50 items.
YOUTUBE_BATCH_SIZE = 50


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


def resolve_channel_id(channel_input):
    """Convert a channel ID or YouTube channel URL into a channel ID."""

    channel_input = channel_input.strip()

    # Direct channel ID
    if channel_input.startswith("UC") and "/" not in channel_input:
        return channel_input

    # Channel URL: youtube.com/channel/UC...
    if "/channel/" in channel_input:
        return channel_input.split("/channel/")[1].split("/")[0]

    # Handle URL: youtube.com/@username
    if "/@" in channel_input:
        handle = "@" + channel_input.split("/@")[1].split("/")[0]

        youtube = get_youtube_client()

        response = youtube.channels().list(
            part="id",
            forHandle=handle
        ).execute()

        if response["items"]:
            return response["items"][0]["id"]

    return None


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


def get_all_upload_video_ids(youtube, uploads_playlist_id):
    """
    Retrieve every video ID available from a channel's uploads playlist.

    YouTube returns at most 50 playlist items in one API response. When more
    videos exist, the response contains nextPageToken. We keep sending another
    request with that token until YouTube stops returning a nextPageToken.
    """

    video_ids = []
    next_page_token = None

    while True:
        # Ask for the maximum number YouTube allows on each playlist request.
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=YOUTUBE_BATCH_SIZE,
            pageToken=next_page_token,
        )

        response = request.execute()

        # Add every valid video ID from this page to our master list.
        for item in response.get("items", []):
            video_id = item.get("contentDetails", {}).get("videoId")
            if video_id:
                video_ids.append(video_id)

        # YouTube gives us this token only when another page exists.
        next_page_token = response.get("nextPageToken")

        # No token means we reached the end of the uploads playlist.
        if not next_page_token:
            break

    return video_ids


def get_video_details_in_batches(youtube, video_ids):
    """
    Retrieve detailed metadata/statistics for all collected video IDs.

    The video IDs are processed in groups of 50. A channel can therefore have
    far more than 50 videos even though each individual API request is small.
    """

    videos_by_id = {}

    for start in range(0, len(video_ids), YOUTUBE_BATCH_SIZE):
        # Example for 120 IDs:
        # batch 1 = IDs 0-49
        # batch 2 = IDs 50-99
        # batch 3 = IDs 100-119
        batch_ids = video_ids[start : start + YOUTUBE_BATCH_SIZE]

        response = (
            youtube.videos()
            .list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch_ids),
            )
            .execute()
        )

        for video in response.get("items", []):
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

    return videos_by_id


def get_all_videos(channel_id):
    """Retrieve all publicly available uploaded videos and their statistics."""

    # First get the channel so we can find its uploads playlist ID.
    channel = get_channel_by_id(channel_id)
    if channel is None:
        return None

    uploads_playlist_id = channel.get("uploads_playlist_id")
    if not uploads_playlist_id:
        return {
            "channel": channel,
            "videos": [],
            "retrieved_video_count": 0,
        }

    youtube = get_youtube_client()

    # Step 1: follow YouTube pagination until every available upload ID is read.
    video_ids = get_all_upload_video_ids(
        youtube=youtube,
        uploads_playlist_id=uploads_playlist_id,
    )

    if not video_ids:
        return {
            "channel": channel,
            "videos": [],
            "retrieved_video_count": 0,
        }

    # Step 2: retrieve full details/statistics in batches of 50 video IDs.
    videos_by_id = get_video_details_in_batches(youtube, video_ids)

    # Keep the same newest-to-oldest order returned by the uploads playlist.
    # A video can be missing here if YouTube returned its playlist item but did
    # not return that video from videos.list (for example, unavailable content).
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
