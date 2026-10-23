"""
Single access point for all Gemini API calls in NEURO-MAS.

No other module should import google.genai directly -- always go through
generate_structured() or embed_text() here. This keeps model selection,
retries, timeouts, and logging (latency + token counts, NEVER raw prompt
content, since prompts may contain real patient symptom data) in exactly
one place.
"""
from __future__ import annotations

import logging
import time
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core import privacy
from app.core.config import get_settings

logger = logging.getLogger("neuromas.gemini")

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None

# Per-request network timeout (ms). Audit fix: there was none, so a hung connection could stall a
# workflow (and therefore an emergency response) indefinitely.
REQUEST_TIMEOUT_MS = 30_000
_TRANSIENT_HTTP_CODES = {408, 429, 500, 502, 503, 504}


def is_transient_error(exc: BaseException) -> bool:
    """Retry only failures that can plausibly succeed on retry. Audit fix: previously EVERY exception
    (bad API key, 400 invalid request, programming errors) was retried 3x with backoff, delaying the
    fallback path by ~3 s for errors that can never succeed."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code in _TRANSIENT_HTTP_CODES
    try:
        import httpx
        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, (ConnectionError, TimeoutError))


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set -- check backend/.env")
        _client = genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    return _client


class GeminiServiceError(Exception):
    """Raised when Gemini fails after all retries. Callers must decide their
    own fallback behavior -- this module never silently returns fabricated
    data when the real call fails."""


class ExternalProcessingBlockedError(GeminiServiceError):
    """Policy/consent refused the call BEFORE anything was sent. Subclasses GeminiServiceError so every existing
    caller falls back to its deterministic template exactly as for an outage."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(is_transient_error),
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

    decision, (prompt, system_instruction) = privacy.authorize_external_call(
        "generate", [prompt, system_instruction], model=chosen_model)
    if not decision.allowed:
        logger.info("gemini_call_blocked reason=%s", decision.reason)
        raise ExternalProcessingBlockedError(f"external processing blocked: {decision.reason}")

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(is_transient_error),
    reraise=True,
)
def _call_embed(model: str, content: str, dimensions: int):
    client = _get_client()
    return client.models.embed_content(
        model=model,
        contents=content,
        config=types.EmbedContentConfig(output_dimensionality=dimensions),
    )


def embed_text(text: str, *, task_hint: str | None = None) -> list[float]:
    """
    Turns text into a vector embedding for semantic search (RAG). Retries
    transient failures the same way generate_structured does. Raises
    GeminiServiceError on persistent failure -- callers must handle this
    (e.g. skip semantic retrieval and fall back to the lexical matcher)
    rather than crash.

    task_hint mirrors Gemini's recommended prompt-prefix convention for its
    embedding model: use "search document" when embedding knowledge base
    entries at ingestion time, "search query" when embedding a patient's
    symptom text at lookup time. Using consistent hints on both sides
    matters for retrieval quality.
    """
    settings = get_settings()
    decision, (text,) = privacy.authorize_external_call("embed", [text], model=settings.gemini_embedding_model)
    if not decision.allowed:
        logger.info("gemini_embed_blocked reason=%s", decision.reason)
        raise ExternalProcessingBlockedError(f"external processing blocked: {decision.reason}")
    content = f"task: {task_hint} | query: {text}" if task_hint else text

    started = time.monotonic()
    try:
        result = _call_embed(settings.gemini_embedding_model, content, settings.gemini_embedding_dimensions)
    except Exception as exc:
        elapsed_ms = round((time.monotonic() - started) * 1000)
        logger.error(
            "gemini_embed_failed model=%s elapsed_ms=%d error=%s",
            settings.gemini_embedding_model, elapsed_ms, repr(exc),
        )
        raise GeminiServiceError(f"Gemini embedding call failed after retries: {exc}") from exc

    elapsed_ms = round((time.monotonic() - started) * 1000)
    vector = result.embeddings[0].values
    logger.info(
        "gemini_embed_ok model=%s elapsed_ms=%d dimensions=%d",
        settings.gemini_embedding_model, elapsed_ms, len(vector),
    )
    return vector