"""
External-processing policy, consent gate and redaction for every call that leaves the system (Phase 2, F3).

All Gemini traffic goes through app/ai/gemini_service.py; that module calls `authorize_external_call(...)` before
sending anything. The decision depends on

  * EXTERNAL_PROCESSING_MODE  disabled | redacted (default) | unredacted
  * REQUIRE_PATIENT_CONSENT   default FALSE since 2026-10-24 (owner decision: AI always on); when true, a patient-scoped call is blocked unless that patient's record says
                              `external_ai_consent = true` (default false; fail closed)
  * the ProcessingContext     set by whoever knows the patient (orchestrator, chat, symptom-assessment route)

A call made with NO context is blocked while consent is required, so a forgotten `processing_scope(...)` cannot leak
data silently; non-patient work (e.g. knowledge-base ingestion) must say so with `system_scope(...)`.

Redaction is BEST-EFFORT pattern masking (emails, URLs, SSN-shaped numbers, phone numbers, dates, long numeric IDs and
the patient's own known name parts). It cannot find an arbitrary name, address or employer typed into free text, and it
is not de-identification under any legal standard. Nothing here is a compliance claim.

Audit: each decision appends {purpose, kind, allowed, reason, mode, redactions(counts only), chars} to the scope's
event list - never the text itself.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
from dataclasses import dataclass, field
from typing import Iterator

from app.core.config import get_settings

MODES = ("disabled", "redacted", "unredacted")


@dataclass
class ProcessingContext:
    purpose: str
    consent: bool | None = None            # None = not a patient-scoped call
    system: bool = False                   # explicitly non-patient (e.g. KB ingestion)
    identifiers: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


@dataclass
class Decision:
    allowed: bool
    reason: str
    mode: str
    redact: bool


_ctx: contextvars.ContextVar[ProcessingContext | None] = contextvars.ContextVar("neuromas_processing_ctx", default=None)


@contextlib.contextmanager
def processing_scope(*, purpose: str, consent: bool | None, identifiers: list[str] | None = None) -> Iterator[ProcessingContext]:
    ctx = ProcessingContext(purpose=purpose, consent=bool(consent), identifiers=[i for i in (identifiers or []) if i])
    token = _ctx.set(ctx)
    try:
        yield ctx
    finally:
        _ctx.reset(token)


@contextlib.contextmanager
def system_scope(purpose: str) -> Iterator[ProcessingContext]:
    ctx = ProcessingContext(purpose=purpose, system=True)
    token = _ctx.set(ctx)
    try:
        yield ctx
    finally:
        _ctx.reset(token)


def current_context() -> ProcessingContext | None:
    return _ctx.get()


def decide(kind: str) -> Decision:
    s = get_settings()
    mode = str(getattr(s, "external_processing_mode", "redacted")).lower()
    if mode not in MODES:
        return Decision(False, f"invalid_external_processing_mode:{mode}", mode, True)     # unknown policy -> closed
    if mode == "disabled":
        return Decision(False, "external_processing_disabled_by_policy", mode, True)
    ctx = current_context()
    if getattr(s, "require_patient_consent", False):
        if ctx is None:
            return Decision(False, "no_processing_scope: caller did not declare patient consent or a system purpose", mode, True)
        if not ctx.system and not ctx.consent:
            return Decision(False, "patient_consent_not_recorded", mode, True)
    return Decision(True, "allowed", mode, mode == "redacted")


# --------------------------------------------------------------------------------------------- redaction
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("url", re.compile(r"https?://\S+|www\.\S+")),
    ("ssn_like", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("date", re.compile(r"(?<!\d)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?!\d)")),
    ("phone", re.compile(r"(?<![\d.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?![\d.])|\+\d{8,15}")),
    ("long_id", re.compile(r"(?<![\d.])\d{7,}(?![\d.])")),
]


def redact_text(text: str, identifiers: list[str] | None = None) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    out = text or ""
    for label, rx in _PATTERNS:
        out, k = rx.subn(f"[{label.upper()}]", out)
        if k:
            counts[label] = counts.get(label, 0) + k
    # structured patterns first (so an e-mail containing the patient's name is masked whole), then known names, longest first
    names = sorted({t for ident in (identifiers or []) for t in ([ident] + re.split(r"\s+", ident)) if len(t.strip()) >= 3},
                   key=len, reverse=True)
    for n in names:
        out, k = re.subn(rf"(?<!\w){re.escape(n.strip())}(?!\w)", "[PATIENT]", out, flags=re.IGNORECASE)
        if k:
            counts["patient_name"] = counts.get("patient_name", 0) + k
    return out, counts


def authorize_external_call(kind: str, texts: list[str | None], *, model: str = "") -> tuple[Decision, list[str | None]]:
    """Returns the decision and the (possibly redacted) texts to send. Records an audit event with counts only."""
    d = decide(kind)
    ctx = current_context()
    redactions: dict[str, int] = {}
    out: list[str | None] = list(texts)
    if d.allowed and d.redact:
        out = []
        for t in texts:
            if t is None:
                out.append(None)
                continue
            r, c = redact_text(t, ctx.identifiers if ctx else None)
            out.append(r)
            for k, v in c.items():
                redactions[k] = redactions.get(k, 0) + v
    if ctx is not None:
        ctx.events.append({"kind": kind, "purpose": ctx.purpose, "allowed": d.allowed, "reason": d.reason, "mode": d.mode,
                           "model": model, "redactions": redactions, "chars_sent": sum(len(t) for t in out if t) if d.allowed else 0})
    return d, out
