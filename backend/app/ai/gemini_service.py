"""
Single access point for all Gemini API calls in NEURO-MAS.

No other module should import google.genai directly -- always go through
generate_structured() here. This keeps model selection, retries, timeouts,
and logging (latency + token counts, NEVER raw prompt content, since prompts
may contain real patient symptom data) in exactly one place.
"""
from __future__ import annotations

import logging
import time
from typing import TypeVar

from google import genai
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings

logger = logging.getLogger("neuromas.gemini")

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set -- check backend/.env")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


class GeminiServiceError(Exception):
    """Raised when Gemini fails after all retries. Callers (agents) must
    decide their own fallback behavior -- this function never silently
    returns fabricated data when the real call fails."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_gemini(model: str, contents: str, config: dict):
    client = _get_client()
    return client.models.generate_content(model=model, contents=contents, config=config)


def generate_structured(
    prompt: str,
    response_model: type[T],
    *,
    model: str | None = None,
    system_instruction: str | None = None,
) -> T:
    """
    Calls Gemini and forces its output to match response_model's schema.
    Retries transient failures up to 3 times with backoff. Raises
    GeminiServiceError if it still fails, or if Gemini's output didn't
    actually match the schema -- never returns a guess.
    """
    settings = get_settings()
    chosen_model = model or settings.gemini_default_model

    config: dict = {
        "response_mime_type": "application/json",
        "response_schema": response_model,
    }
    if system_instruction:
        config["system_instruction"] = system_instruction

    started = time.monotonic()
    try:
        response = _call_gemini(chosen_model, prompt, config)
    except Exception as exc:
        elapsed_ms = round((time.monotonic() - started) * 1000)
        logger.error(
            "gemini_call_failed model=%s elapsed_ms=%d error=%s",
            chosen_model, elapsed_ms, repr(exc),
        )
        raise GeminiServiceError(f"Gemini call failed after retries: {exc}") from exc

    elapsed_ms = round((time.monotonic() - started) * 1000)
    usage = getattr(response, "usage_metadata", None)
    logger.info(
        "gemini_call_ok model=%s elapsed_ms=%d prompt_tokens=%s output_tokens=%s total_tokens=%s",
        chosen_model,
        elapsed_ms,
        getattr(usage, "prompt_token_count", "?"),
        getattr(usage, "candidates_token_count", "?"),
        getattr(usage, "total_token_count", "?"),
    )

    if response.parsed is None:
        raise GeminiServiceError("Gemini returned output that didn't match the expected schema")
    return response.parsed
