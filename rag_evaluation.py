"""Small repeatable evaluation harness for the capstone RAG chatbot.

The utility does not contact YouTube. It uses the existing Chroma database and
Gemini configuration to exercise multilingual, arbitrary target-language, and
specific-video behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from backend.database_service import get_cached_channel_data, initialize_database
from backend.llm_service import is_gemini_configured
from backend.rag_service import (
    answer_question,
    channel_requires_embedding_rebuild,
    get_indexed_video_ids,
)
from backend.youtube_service import resolve_channel_id


CASES = [
    ("english_auto", "What are the main topics discussed in this channel?", "auto"),
    ("spanish_auto", "¿Cuáles son los temas principales que se discuten en este canal?", "auto"),
    ("telugu_auto", "ఈ ఛానెల్‌లో చర్చించబడే ప్రధాన విషయాలు ఏమిటి?", "auto"),
    ("english_to_spanish", "What are the main topics discussed in this channel?", "Spanish"),
    ("spanish_to_english", "¿Cuáles son los temas principales que se discuten en este canal?", "English"),
    ("english_to_japanese", "What are the main topics discussed in this channel?", "Japanese"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multilingual RAG behavior.")
    parser.add_argument("channel", help="YouTube UC channel ID or supported channel URL")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    initialize_database()
    channel_id = resolve_channel_id(args.channel)
    if not channel_id:
        print("Error: channel could not be resolved.", file=sys.stderr)
        return 1

    channel_data = get_cached_channel_data(channel_id)
    if channel_data is None:
        print("Error: channel is not cached. Analyze/transcribe it first.", file=sys.stderr)
        return 1

    indexed = get_indexed_video_ids(channel_id)
    if not indexed:
        print("Error: no Chroma content is indexed for this channel.", file=sys.stderr)
        return 1

    if channel_requires_embedding_rebuild(channel_id):
        print(
            "Error: this channel uses an older embedding model. Run "
            "'python rag_builder.py UC_CHANNEL_ID --rebuild' first.",
            file=sys.stderr,
        )
        return 1

    results = []
    for name, question, answer_language in CASES:
        response = answer_question(
            channel_id=channel_id,
            question=question,
            transcripts=[],
            top_k=args.top_k,
            answer_language=answer_language,
        )
        results.append(
            {
                "case": name,
                "question": question,
                "answer_language": answer_language,
                "answer": response.get("answer"),
                "source_video_ids": [s.get("video_id") for s in response.get("sources", [])],
                "retrieved_chunk_count": response.get("retrieved_chunk_count", 0),
            }
        )

    # Specific-video isolation test: choose the newest cached video that is indexed.
    target = next(
        (video for video in channel_data["videos"] if str(video.get("video_id")) in indexed),
        None,
    )
    if target is not None:
        response = answer_question(
            channel_id=channel_id,
            question=f"What is the video '{target.get('title')}' about?",
            transcripts=[],
            top_k=args.top_k,
            target_video_id=str(target["video_id"]),
            answer_language="auto",
        )
        source_ids = [s.get("video_id") for s in response.get("sources", [])]
        results.append(
            {
                "case": "specific_video_filter",
                "target_video_id": target["video_id"],
                "target_title": target.get("title"),
                "answer": response.get("answer"),
                "source_video_ids": source_ids,
                "passed_source_isolation": bool(source_ids) and set(source_ids) == {str(target["video_id"])},
            }
        )

    output = {
        "channel_id": channel_id,
        "gemini_configured": is_gemini_configured(),
        "note": (
            "Review multilingual answers for language correctness. If Gemini is not configured, "
            "the evaluation can only exercise retrieval/fallback behavior."
        ),
        "results": results,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"Channel: {channel_id}")
        print(f"Gemini configured: {output['gemini_configured']}")
        for item in results:
            print("\n" + "=" * 72)
            print(f"CASE: {item['case']}")
            if item.get("question"):
                print(f"QUESTION: {item['question']}")
            if item.get("answer_language"):
                print(f"ANSWER LANGUAGE: {item['answer_language']}")
            if "passed_source_isolation" in item:
                print(f"SPECIFIC-VIDEO FILTER PASS: {item['passed_source_isolation']}")
            print(f"SOURCES: {item.get('source_video_ids', [])}")
            print(f"ANSWER:\n{item.get('answer', '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
