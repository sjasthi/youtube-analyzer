"""Gemini generation helper for the YouTube Channel Analyzer RAG feature.

This file is intentionally separate from rag_service.py:
- rag_service.py decides which transcript chunks are relevant.
- llm_service.py turns that retrieved context into a readable answer.

The Gemini helper supports an ordered model fallback chain plus short retries for
transient 503/high-demand failures:
- 503/high demand: retry the same model with exponential backoff, then use the
  next model only for the current multi-call generation operation.
- 429/model-access failures: fall back and keep the lower-priority model active
  for the current FastAPI process because those failures are less likely to be
  resolved by immediately retrying the same model.
- authentication, malformed requests, empty responses, and safety/policy blocks
  never use fallback to bypass the underlying problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

# Load the project-root .env file even when this module is tested by itself.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# The primary model and ordered fallback models can be changed in .env without
# editing Python source code.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_FALLBACK_MODELS_RAW = os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite"
).strip()

# 503s are normally temporary provider-capacity failures. Retry the same model
# briefly before moving to the next model. Two retries means at most three total
# attempts on a model (initial + two retries).
GEMINI_503_MAX_RETRIES = max(0, int(os.getenv("GEMINI_503_MAX_RETRIES", "2")))
GEMINI_503_RETRY_BASE_SECONDS = max(
    0.0, float(os.getenv("GEMINI_503_RETRY_BASE_SECONDS", "1.0"))
)


def _build_model_chain() -> tuple[str, ...]:
    """Return the primary model followed by unique fallbacks in configured order."""
    configured = [GEMINI_MODEL]
    configured.extend(
        model.strip()
        for model in GEMINI_FALLBACK_MODELS_RAW.split(",")
        if model.strip()
    )

    unique: list[str] = []
    for model in configured:
        if model and model not in unique:
            unique.append(model)
    return tuple(unique)


GEMINI_MODEL_CHAIN = _build_model_chain()
_active_model = GEMINI_MODEL_CHAIN[0]
_active_model_reason: str | None = None
_model_state_lock = threading.Lock()


@dataclass
class GeminiGenerationSession:
    """State shared by all Gemini calls belonging to one logical operation.

    Channel Summary makes several LLM calls. If 3.8 returns repeated 503s, this
    session remembers the temporary 3.5 fallback so the remaining batches do
    not keep hammering 3.8. A new Channel Summary creates a new session and will
    try the globally preferred model again.
    """

    temporary_model: str | None = None
    temporary_reason: str | None = None
    temporary_from_model: str | None = None


def create_gemini_generation_session() -> GeminiGenerationSession:
    """Create isolated temporary-fallback state for one multi-call operation."""
    return GeminiGenerationSession()


def is_gemini_configured() -> bool:
    """Return True when a Gemini API key is available to the application."""
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def _status_code_from_error(error: Exception) -> int | None:
    """Best-effort extraction of an HTTP-like status code from SDK exceptions."""
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        if isinstance(value, int):
            return value

        # Some SDK exceptions expose enum-like values such as StatusCode(429).
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, int):
            return enum_value

    return None


def _error_message(error: Exception) -> str:
    return f"{type(error).__name__}: {error}".lower()


def _is_safety_or_policy_error(error: Exception) -> bool:
    message = _error_message(error)
    return any(
        marker in message
        for marker in ("safety", "policy violation", "prohibited content")
    )


def _is_transient_503(error: Exception) -> bool:
    """Return True for temporary provider-capacity/unavailable failures."""
    if _is_safety_or_policy_error(error):
        return False
    status_code = _status_code_from_error(error)
    message = _error_message(error)
    if status_code == 503:
        return True
    return any(
        marker in message
        for marker in (
            "service unavailable",
            "temporarily unavailable",
            "high demand",
            "overloaded",
            "status': 'unavailable",
            'status": "unavailable',
        )
    )


def _is_sticky_fallback_error(error: Exception) -> bool:
    """Return True when a process-level downgrade is reasonable.

    429/rate/quota limits and explicit model access/not-found errors are treated
    as sticky. 503 is deliberately excluded because it is usually temporary.
    """
    if _is_safety_or_policy_error(error) or _is_transient_503(error):
        return False

    status_code = _status_code_from_error(error)
    message = _error_message(error)

    if status_code == 429:
        return True
    if any(
        marker in message
        for marker in (
            "resource_exhausted",
            "too many requests",
            "rate limit",
            "quota exceeded",
        )
    ):
        return True

    model_access_markers = (
        "model not found",
        "model is not found",
        "model unavailable",
        "model is unavailable",
        "model is not available",
        "no longer available",
        "not available to new users",
        "does not have access to model",
        "access to this model",
        "unsupported model",
    )
    if status_code in (403, 404) and any(marker in message for marker in model_access_markers):
        return True
    return any(marker in message for marker in model_access_markers)


def _should_fallback(error: Exception) -> bool:
    """Return True only when trying another configured model can reasonably help."""
    return _is_transient_503(error) or _is_sticky_fallback_error(error)


def _current_model() -> str:
    with _model_state_lock:
        return _active_model


def _set_active_model(model: str, reason: str | None = None) -> None:
    global _active_model, _active_model_reason
    with _model_state_lock:
        _active_model = model
        _active_model_reason = reason


def _model_index(model: str | None) -> int | None:
    if not model:
        return None
    try:
        return GEMINI_MODEL_CHAIN.index(model)
    except ValueError:
        return None


def _models_to_try(session: GeminiGenerationSession | None = None) -> tuple[str, ...]:
    """Return the correct model slice for global and operation-scoped state."""
    current = _current_model()
    global_index = _model_index(current)
    if global_index is None:
        global_index = 0

    start_index = global_index
    if session is not None:
        temporary_index = _model_index(session.temporary_model)
        if temporary_index is not None:
            # Never move upward past a process-level sticky downgrade. Otherwise,
            # honor the session's temporary 503 fallback for the rest of this
            # summary/comparison operation.
            start_index = max(global_index, temporary_index)

    return GEMINI_MODEL_CHAIN[start_index:]


def _fallback_warning_for(
    model: str,
    session: GeminiGenerationSession | None = None,
) -> str | None:
    """Return a user-facing explanation when a lower-priority model was used."""
    if model == GEMINI_MODEL:
        return None

    if session is not None and session.temporary_model:
        temporary_index = _model_index(session.temporary_model)
        model_index = _model_index(model)
        if temporary_index is not None and model_index is not None and model_index >= temporary_index:
            from_model = session.temporary_from_model or GEMINI_MODEL
            reason = session.temporary_reason or "temporary provider overload"
            return (
                f"Gemini temporarily fell back from {from_model} to {model} for this operation "
                f"after repeated temporary availability errors ({reason}). The next new "
                f"Channel Summary will try {GEMINI_MODEL} again unless a process-level "
                "rate/access downgrade is active."
            )

    with _model_state_lock:
        reason = _active_model_reason

    warning = f"Gemini automatically downgraded from {GEMINI_MODEL} to {model}"
    if reason:
        warning += f" because the higher-priority model hit a rate/access limit ({reason})."
    else:
        warning += " because the higher-priority model hit a rate/access limit."
    return warning


def _generate_with_model_chain(
    client: Any,
    types_module: Any,
    prompt: str,
    *,
    session: GeminiGenerationSession | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, str | bool | None]:
    """Generate text with retry + model fallback behavior.

    503/high-demand errors are retried on the same model using exponential
    backoff. If retries are exhausted, the next model is used temporarily for
    the supplied session. 429/model-access errors immediately move down the
    chain and make that lower model process-level sticky.
    """
    if session is None:
        session = GeminiGenerationSession()

    models = _models_to_try(session)
    attempted_models: list[str] = []
    attempt_counts: dict[str, int] = {}
    last_error: Exception | None = None

    for index, model in enumerate(models):
        attempted_models.append(model)
        max_attempts = 1 + GEMINI_503_MAX_RETRIES

        for attempt_number in range(1, max_attempts + 1):
            attempt_counts[model] = attempt_counts.get(model, 0) + 1
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types_module.GenerateContentConfig(
                        thinking_config=types_module.ThinkingConfig(thinking_level="low")
                    ),
                )
            except Exception as error:
                last_error = error

                if _is_transient_503(error):
                    if attempt_number < max_attempts:
                        delay = GEMINI_503_RETRY_BASE_SECONDS * (2 ** (attempt_number - 1))
                        logger.warning(
                            "Gemini model %s returned temporary 503/unavailable (attempt %s/%s). "
                            "Retrying after %.2fs. Error: %s",
                            model,
                            attempt_number,
                            max_attempts,
                            delay,
                            error,
                        )
                        if delay > 0:
                            sleep_fn(delay)
                        continue

                    has_fallback = index < len(models) - 1
                    if not has_fallback:
                        break

                    next_model = models[index + 1]
                    reason = str(error).strip() or type(error).__name__
                    logger.warning(
                        "Gemini model %s remained temporarily unavailable after %s attempts; "
                        "using %s for the rest of this generation operation. Error: %s",
                        model,
                        max_attempts,
                        next_model,
                        reason,
                    )
                    session.temporary_model = next_model
                    session.temporary_reason = reason
                    session.temporary_from_model = model
                    break

                if _is_sticky_fallback_error(error):
                    has_fallback = index < len(models) - 1
                    if not has_fallback:
                        break

                    next_model = models[index + 1]
                    reason = str(error).strip() or type(error).__name__
                    logger.warning(
                        "Gemini model %s hit a rate/access fallback error; switching process "
                        "preference to %s. Error: %s",
                        model,
                        next_model,
                        reason,
                    )
                    _set_active_model(next_model, reason)
                    # Keep this operation aligned with the sticky downgrade too.
                    session.temporary_model = next_model
                    session.temporary_reason = None
                    session.temporary_from_model = None
                    break

                # Authentication, malformed request, safety/policy, and unrelated
                # failures are not candidates for model fallback.
                break

            else:
                answer = (getattr(response, "text", None) or "").strip()
                if not answer:
                    raise RuntimeError(f"Gemini model {model} returned an empty answer.")

                return {
                    "answer": answer,
                    "provider": "gemini",
                    "model": model,
                    "primary_model": GEMINI_MODEL,
                    "fallback_used": model != GEMINI_MODEL,
                    "fallback_warning": _fallback_warning_for(model, session),
                }

        # The inner loop exits on exhausted transient retries or a sticky
        # fallback error. Continue only if another model is actually allowed.
        if last_error is not None and _should_fallback(last_error) and index < len(models) - 1:
            continue
        break

    attempted_parts = [
        f"{model} ({attempt_counts.get(model, 0)} attempt{'s' if attempt_counts.get(model, 0) != 1 else ''})"
        for model in attempted_models
    ]
    attempted = " -> ".join(attempted_parts) or GEMINI_MODEL
    if last_error is None:
        raise RuntimeError(f"Gemini generation failed. Attempted models: {attempted}.")
    raise RuntimeError(
        f"Gemini generation failed after trying {attempted}. Last error: {last_error}"
    ) from last_error


def generate_with_gemini(
    prompt: str,
    *,
    session: GeminiGenerationSession | None = None,
) -> dict[str, str | bool | None]:
    """Send a completed RAG prompt to Gemini and return generated text."""
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

    client = genai.Client(api_key=api_key)
    return _generate_with_model_chain(client, types, prompt, session=session)
