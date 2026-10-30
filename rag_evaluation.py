"""Repeatable evaluation harness for the capstone RAG chatbot.

This utility never retrieves YouTube transcripts. It evaluates the existing
SQLite/Chroma/Gemini pipeline and prints explicit PASS/FAIL checks for retrieval,
metadata scopes, multilingual generation availability, and source isolation.
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.database_service import get_cached_channel_data, initialize_database
from backend.llm_service import is_gemini_configured
from backend.rag_service import answer_question, channel_requires_embedding_rebuild, get_indexed_video_ids
from backend.youtube_service import resolve_channel_id

CASES = [
    ("english_auto", "What are the main topics discussed in this channel?", "auto"),
    ("spanish_auto", "¿Cuáles son los temas principales que se discuten en este canal?", "auto"),
    ("telugu_auto", "ఈ ఛానెల్‌లో చర్చించబడే ప్రధాన విషయాలు ఏమిటి?", "auto"),
    ("english_to_spanish", "What are the main topics discussed in this channel?", "Spanish"),
    ("spanish_to_english", "¿Cuáles son los temas principales que se discuten en este canal?", "English"),
    ("english_to_japanese", "What are the main topics discussed in this channel?", "Japanese"),
]


def _result(name: str, passed: bool, **details):
    return {"case": name, "passed": bool(passed), **details}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the existing RAG database.")
    parser.add_argument("channel", help="YouTube UC channel ID or supported channel URL")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true")
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
        print("Error: rebuild Chroma with the current embedding model first.", file=sys.stderr)
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
        source_ids = [s.get("video_id") for s in response.get("sources", [])]
        passed = response.get("retrieved_chunk_count", 0) > 0 and bool(response.get("answer"))
        results.append(_result(
            name,
            passed,
            question=question,
            requested_answer_language=answer_language,
            source_video_ids=source_ids,
            retrieved_chunk_count=response.get("retrieved_chunk_count", 0),
            generation_mode=response.get("mode"),
            answer=response.get("answer"),
            note="Language fluency/correctness still requires human review.",
        ))

    indexed_videos = [v for v in channel_data["videos"] if str(v.get("video_id")) in indexed]
    newest = indexed_videos[0] if indexed_videos else None
    if newest:
        response = answer_question(
            channel_id=channel_id,
            question="What is the most recent indexed video about?",
            transcripts=[],
            top_k=args.top_k,
            target_video_id=str(newest["video_id"]),
        )
        source_ids = {str(s.get("video_id")) for s in response.get("sources", [])}
        results.append(_result(
            "latest_video_source_isolation",
            bool(source_ids) and source_ids == {str(newest["video_id"])},
            expected_video_id=newest["video_id"],
            source_video_ids=sorted(source_ids),
        ))

        response = answer_question(
            channel_id=channel_id,
            question=f"What is the video '{newest.get('title')}' about?",
            transcripts=[],
            top_k=args.top_k,
            target_video_id=str(newest["video_id"]),
        )
        source_ids = {str(s.get("video_id")) for s in response.get("sources", [])}
        results.append(_result(
            "specific_video_filter",
            bool(source_ids) and source_ids == {str(newest["video_id"])},
            expected_video_id=newest["video_id"],
            source_video_ids=sorted(source_ids),
        ))

    if len(indexed_videos) >= 2:
        a, b = indexed_videos[:2]
        expected = {str(a["video_id"]), str(b["video_id"])}
        response = answer_question(
            channel_id=channel_id,
            question=f"Compare '{a.get('title')}' and '{b.get('title')}'.",
            transcripts=[],
            top_k=max(2, args.top_k),
            target_video_ids=list(expected),
        )
        source_ids = {str(s.get("video_id")) for s in response.get("sources", [])}
        results.append(_result(
            "two_video_comparison_sources",
            expected.issubset(source_ids),
            expected_video_ids=sorted(expected),
            source_video_ids=sorted(source_ids),
        ))

    if len(indexed_videos) >= 6:
        comparison_videos = indexed_videos[: min(10, len(indexed_videos))]
        expected_ids = [str(video["video_id"]) for video in comparison_videos]
        response = answer_question(
            channel_id=channel_id,
            question=f"Compare these {len(expected_ids)} indexed videos.",
            transcripts=[],
            target_video_ids=expected_ids,
            comparison_mode=True,
        )
        source_ids = {str(s.get("video_id")) for s in response.get("sources", [])}
        results.append(_result(
            "hierarchical_multi_video_comparison",
            set(expected_ids).issubset(source_ids),
            expected_video_count=len(expected_ids),
            source_video_count=len(source_ids),
            comparison_strategy=response.get("comparison_strategy"),
            comparison_batch_count=response.get("comparison_batch_count"),
            generation_mode=response.get("mode"),
        ))

    passed_count = sum(1 for item in results if item["passed"])
    output = {
        "channel_id": channel_id,
        "gemini_configured": is_gemini_configured(),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "total": len(results),
        "results": results,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"Channel: {channel_id}")
        print(f"Gemini configured: {output['gemini_configured']}")
        print(f"Automated checks: {passed_count}/{len(results)} passed")
        for item in results:
            print("\n" + "=" * 72)
            print(f"{'PASS' if item['passed'] else 'FAIL'}: {item['case']}")
            if item.get("source_video_ids") is not None:
                print(f"Sources: {item['source_video_ids']}")
            if item.get("answer"):
                print(f"Answer:\n{item['answer']}")
            if item.get("note"):
                print(f"Note: {item['note']}")

    return 0 if output["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
