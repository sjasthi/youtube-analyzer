"""Standalone RAG database builder.

This utility processes transcripts already saved in SQLite and builds the local
Chroma vector database. It does not retrieve transcripts from YouTube.

Examples:
    python rag_builder.py UCxxxxxxxx --limit 0
    python rag_builder.py UCxxxxxxxx --rebuild
"""

from __future__ import annotations

import argparse
import sys

from backend.database_service import get_cached_channel_data, initialize_database
from backend.rag_service import (
    EMBEDDING_MODEL_NAME,
    channel_requires_embedding_rebuild,
    delete_channel_index,
    get_rag_storage_stats,
    index_transcripts,
)
from backend.transcript_service import get_cached_video_transcripts
from backend.youtube_service import resolve_channel_id


def _select_videos(videos: list[dict], limit: int) -> list[dict]:
    if limit < 0:
        raise ValueError("--limit must be 0 or a positive integer.")
    return videos if limit == 0 else videos[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Chroma from transcripts already cached in SQLite."
    )
    parser.add_argument("channel", help="YouTube UC channel ID or supported channel URL")
    parser.add_argument("--limit", type=int, default=0, help="Newest N videos; 0 = all.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete this channel's Chroma chunks and rebuild them from SQLite.",
    )
    args = parser.parse_args()

    initialize_database()

    try:
        channel_id = resolve_channel_id(args.channel)
        if not channel_id:
            raise RuntimeError("Channel could not be resolved.")

        channel_data = get_cached_channel_data(channel_id)
        if channel_data is None:
            raise RuntimeError(
                "No cached channel metadata exists. Run youtube_transcriptor.py first."
            )

        videos = _select_videos(channel_data["videos"], args.limit)
        cached = get_cached_video_transcripts(videos)
        transcripts = cached["transcripts"]

        if not transcripts:
            raise RuntimeError(
                "No cached transcripts are available. Run youtube_transcriptor.py first."
            )

        rebuild_needed = channel_requires_embedding_rebuild(channel_id)
        if args.rebuild or rebuild_needed:
            delete_channel_index(channel_id)
            reason = "requested" if args.rebuild else "embedding model changed"
            print(f"Rebuilding existing Chroma index ({reason}).")

        summary = index_transcripts(
            channel_id,
            transcripts,
            channel_title=channel_data["channel"].get("title"),
        )
        stats = get_rag_storage_stats(channel_id)

    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("\nRAG build complete")
    print(f"  Channel: {channel_data['channel'].get('title') or channel_id}")
    print(f"  Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  Cached transcripts supplied: {len(transcripts)}")
    print(f"  Missing SQLite transcripts: {len(cached['missing_video_ids'])}")
    print(f"  Newly indexed videos: {summary['indexed_video_count']}")
    print(f"  Newly indexed chunks: {summary['indexed_chunk_count']}")
    print(f"  Total indexed videos: {stats.get('video_count', 0)}")
    print(f"  Total Chroma chunks: {stats.get('chunk_count', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
