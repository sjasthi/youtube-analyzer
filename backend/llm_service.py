"""Gemini generation helper for the YouTube Channel Analyzer RAG feature.

This file is intentionally separate from rag_service.py:
- rag_service.py decides which transcript chunks are relevant.
- llm_service.py turns that retrieved context into a readable answer.

Keeping those responsibilities separate makes it easy to change the generation
provider later without rewriting transcript chunking or FAISS retrieval.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# Load the project-root .env file even when this module is tested by itself.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# The model can be changed in .env without editing Python source code.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def is_gemini_configured() -> bool:
    """Return True when a Gemini API key is available to the application."""
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def generate_with_gemini(prompt: str) -> dict[str, str]:
    """Send a completed RAG prompt to Gemini and return generated text.

    The API key is read from the environment so it never needs to be hard-coded
    in the project or committed to GitHub.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to your .env file to "
            "enable generated RAG answers."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError(
            "Gemini dependency is missing. Install backend/requirements.txt."
        ) from error

    try:
        client = genai.Client(api_key=api_key)

        # Low thinking is sufficient for answering from already-retrieved
        # transcript context and keeps latency/resource usage lower.
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
            ),
        )
    except Exception as error:
        raise RuntimeError(f"Gemini generation failed: {error}") from error

    answer = (getattr(response, "text", None) or "").strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty answer.")

    return {
        "answer": answer,
        "provider": "gemini",
        "model": GEMINI_MODEL,
    }
