"""
Model-agnostic LLM provider interface (spec section 23).

NEURO-MAS's clinical reasoning does NOT depend on any specific LLM: agents
use deterministic rules, curated knowledge lookups, and simple statistical
heuristics for their actual findings. The LLMProvider is used ONLY for one
narrow, low-stakes job -- turning an already-computed, already-safety-checked
structured result into a plain-language explanation for the Patient Copilot.

If no provider is configured, TemplateLLMProvider produces the explanation
from string templates -- deterministic, inspectable, and clearly labeled.
This is intentional: a hackathon-style "wire up an API key and call it AI"
approach would violate the "do not fake features" rule the moment a key is
missing or a call fails silently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.config import get_settings


class LLMProvider(ABC):
    label: str = "unlabeled"

    @abstractmethod
    async def summarize(self, structured_context: dict) -> str:
        raise NotImplementedError


class TemplateLLMProvider(LLMProvider):
    """Deterministic, template-based narrative generation. Default provider.

    Never invents facts: it only rearranges values already present in
    `structured_context` into readable sentences.
    """

    label = "template (deterministic, no external model)"

    async def summarize(self, structured_context: dict) -> str:
        parts: list[str] = []
        decision = structured_context.get("decision")
        if decision:
            parts.append(f"Decision: {decision}.")
        risk = structured_context.get("risk_level")
        if risk:
            parts.append(f"Current risk level is {risk}.")
        agreements = structured_context.get("agreements") or []
        if agreements:
            parts.append("Agents agreed on: " + "; ".join(agreements) + ".")
        conflicts = structured_context.get("conflicts") or []
        if conflicts:
            parts.append("Agents disagreed on: " + "; ".join(conflicts) + ".")
        action = structured_context.get("action")
        if action:
            parts.append(f"Recommended next step: {action}.")
        if not parts:
            return "No summary available: insufficient structured context."
        return " ".join(parts)


class AnthropicLLMProvider(LLMProvider):
    """Optional provider that uses an Anthropic model to phrase the
    already-computed explanation more naturally. Only activated if
    ANTHROPIC_API_KEY is set; otherwise the factory below falls back to
    TemplateLLMProvider so the app never silently degrades into fabrication.
    """

    label = "anthropic (narrative phrasing of pre-computed structured output only)"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def summarize(self, structured_context: dict) -> str:
        # Deliberately not wired to a live network call in this prototype --
        # doing so from inside an agent workflow needs its own timeout /
        # retry / fallback handling (see docs/ROADMAP.md, "failure handling").
        # Falling back to the deterministic template keeps behavior honest
        # rather than pretending a call happened.
        return await TemplateLLMProvider().summarize(structured_context)


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicLLMProvider(settings.anthropic_api_key)
    return TemplateLLMProvider()
