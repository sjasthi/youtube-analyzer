"""Unit tests for Gemini retry and model fallback behavior.

These tests use fakes and do not make network/API calls.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend import llm_service


class FakeGeminiError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls: list[str] = []

    def generate_content(self, *, model, contents, config):
        del contents, config
        self.calls.append(model)
        outcome = self.outcomes[model]
        if isinstance(outcome, list):
            if not outcome:
                raise AssertionError(f"No fake outcomes remain for {model}")
            outcome = outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


class FakeTypes:
    class ThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


class GeminiFallbackTests(unittest.TestCase):
    def setUp(self):
        self.original_chain = llm_service.GEMINI_MODEL_CHAIN
        self.original_primary = llm_service.GEMINI_MODEL
        self.original_active = llm_service._current_model()
        self.original_reason = llm_service._active_model_reason
        self.original_retries = llm_service.GEMINI_503_MAX_RETRIES
        self.original_base_delay = llm_service.GEMINI_503_RETRY_BASE_SECONDS

        llm_service.GEMINI_MODEL = "primary-model"
        llm_service.GEMINI_MODEL_CHAIN = (
            "primary-model",
            "fallback-model",
            "last-model",
        )
        llm_service.GEMINI_503_MAX_RETRIES = 2
        llm_service.GEMINI_503_RETRY_BASE_SECONDS = 1.0
        llm_service._set_active_model("primary-model", None)

    def tearDown(self):
        llm_service.GEMINI_MODEL = self.original_primary
        llm_service.GEMINI_MODEL_CHAIN = self.original_chain
        llm_service.GEMINI_503_MAX_RETRIES = self.original_retries
        llm_service.GEMINI_503_RETRY_BASE_SECONDS = self.original_base_delay
        llm_service._set_active_model(self.original_active, self.original_reason)

    def test_503_retries_same_model_before_fallback(self):
        models = FakeModels(
            {
                "primary-model": [
                    FakeGeminiError("high demand", 503),
                    FakeGeminiError("high demand", 503),
                    "primary recovered",
                ],
                "fallback-model": "should not be used",
                "last-model": "should not be used",
            }
        )
        delays: list[float] = []
        client = SimpleNamespace(models=models)

        result = llm_service._generate_with_model_chain(
            client,
            FakeTypes,
            "prompt",
            sleep_fn=delays.append,
        )

        self.assertEqual(result["model"], "primary-model")
        self.assertEqual(models.calls, ["primary-model"] * 3)
        self.assertEqual(delays, [1.0, 2.0])
        self.assertEqual(llm_service._current_model(), "primary-model")

    def test_503_fallback_is_temporary_for_one_session(self):
        models = FakeModels(
            {
                "primary-model": [FakeGeminiError("high demand", 503)] * 3,
                "fallback-model": ["fallback batch 1", "fallback batch 2"],
                "last-model": "last answer",
            }
        )
        client = SimpleNamespace(models=models)
        session = llm_service.create_gemini_generation_session()

        first = llm_service._generate_with_model_chain(
            client, FakeTypes, "prompt", session=session, sleep_fn=lambda _: None
        )

        self.assertEqual(first["model"], "fallback-model")
        self.assertTrue(first["fallback_used"])
        self.assertIn("temporarily fell back", first["fallback_warning"])
        self.assertEqual(
            models.calls,
            ["primary-model", "primary-model", "primary-model", "fallback-model"],
        )
        # 503 does not permanently change the process-level preferred model.
        self.assertEqual(llm_service._current_model(), "primary-model")

        models.calls.clear()
        second = llm_service._generate_with_model_chain(
            client, FakeTypes, "prompt 2", session=session, sleep_fn=lambda _: None
        )
        self.assertEqual(second["model"], "fallback-model")
        self.assertEqual(models.calls, ["fallback-model"])

        # A new operation/session tries the primary again.
        models.outcomes["primary-model"] = ["primary works on next summary"]
        models.calls.clear()
        new_session = llm_service.create_gemini_generation_session()
        third = llm_service._generate_with_model_chain(
            client, FakeTypes, "prompt 3", session=new_session, sleep_fn=lambda _: None
        )
        self.assertEqual(third["model"], "primary-model")
        self.assertEqual(models.calls, ["primary-model"])

    def test_429_downgrades_and_sticks_to_fallback(self):
        models = FakeModels(
            {
                "primary-model": FakeGeminiError("RESOURCE_EXHAUSTED", 429),
                "fallback-model": "fallback answer",
                "last-model": "last answer",
            }
        )
        client = SimpleNamespace(models=models)

        result = llm_service._generate_with_model_chain(
            client, FakeTypes, "prompt", sleep_fn=lambda _: None
        )

        self.assertEqual(result["model"], "fallback-model")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(models.calls, ["primary-model", "fallback-model"])
        self.assertEqual(llm_service._current_model(), "fallback-model")

        # A second call should start directly on the process-level fallback.
        models.calls.clear()
        second = llm_service._generate_with_model_chain(
            client, FakeTypes, "prompt 2", sleep_fn=lambda _: None
        )
        self.assertEqual(second["model"], "fallback-model")
        self.assertEqual(models.calls, ["fallback-model"])

    def test_503_can_continue_through_multiple_fallbacks_without_sticky_downgrade(self):
        models = FakeModels(
            {
                "primary-model": [FakeGeminiError("Service unavailable", 503)] * 3,
                "fallback-model": [FakeGeminiError("high demand", 503)] * 3,
                "last-model": "working answer",
            }
        )
        client = SimpleNamespace(models=models)
        session = llm_service.create_gemini_generation_session()

        result = llm_service._generate_with_model_chain(
            client, FakeTypes, "prompt", session=session, sleep_fn=lambda _: None
        )

        self.assertEqual(result["model"], "last-model")
        self.assertEqual(
            models.calls,
            ["primary-model"] * 3 + ["fallback-model"] * 3 + ["last-model"],
        )
        self.assertEqual(llm_service._current_model(), "primary-model")

    def test_bad_request_does_not_fallback(self):
        models = FakeModels(
            {
                "primary-model": FakeGeminiError("Invalid request", 400),
                "fallback-model": "should not be used",
                "last-model": "should not be used",
            }
        )
        client = SimpleNamespace(models=models)

        with self.assertRaises(RuntimeError):
            llm_service._generate_with_model_chain(
                client, FakeTypes, "prompt", sleep_fn=lambda _: None
            )

        self.assertEqual(models.calls, ["primary-model"])
        self.assertEqual(llm_service._current_model(), "primary-model")

    def test_safety_error_does_not_fallback_or_retry(self):
        error = FakeGeminiError("Blocked due to safety policy", 429)
        self.assertFalse(llm_service._should_fallback(error))
        self.assertFalse(llm_service._is_transient_503(error))
        self.assertFalse(llm_service._is_sticky_fallback_error(error))


if __name__ == "__main__":
    unittest.main()
