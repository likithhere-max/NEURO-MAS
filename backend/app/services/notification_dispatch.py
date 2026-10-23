"""
Notification delivery: outbox, state machine, idempotency, bounded retries, escalation (Phase 2, G).

Design
------
A permitted action does not "send" inside the workflow. It becomes a durable `NotificationRecord` (the outbox) and a
dispatcher delivers it after the workflow result is stored. Every record moves through explicit states:

    proposed -> approved -> queued -> sent -> delivered -> acknowledged
                              |         |
                              +--> failed (retries exhausted / permanently rejected) --> escalated

  * sent        the provider ACCEPTED the message (e.g. SMTP 250). Not proof anyone read it.
  * delivered   the provider (or a status poll) CONFIRMED delivery. Only this state may mark an action "executed".
  * acknowledged a human acknowledged it (set by the alert-acknowledge path).
  * escalated   delivery failed for good; the failure was raised through the escalation hook (e.g. a critical alert).

Guarantees (each pinned by tests/test_notification_dispatch.py, using a mock provider - no live delivery is ever made
by the test-suite):
  * idempotency   the key `action_id:channel` is unique; re-dispatching (retry, duplicate request, concurrent call)
                  never sends a second message for a record that is queued/sent/delivered/acknowledged/escalated.
  * bounded retry only transient failures (timeout, exception, provider 'transient_error') are retried, at most
                  `max_attempts` times, with exponential backoff; a permanent rejection is not retried.
  * timeouts      each attempt runs under asyncio.wait_for.
  * no PHI        message bodies are generic; audit events hold ids, states and error CLASSES only (never recipient
                  addresses, bodies or provider error text).
  * truthful      nothing is called "executed" unless the record reached `delivered`.

No live provider is enabled unless NOTIFICATION_CHANNEL is configured by the owner. `SmtpEmailProvider` is real code
but has been verified only against a stand-in for smtplib; it has never sent a real message from this repository.
"""
from __future__ import annotations

import asyncio
import smtplib
import ssl
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Awaitable, Callable, Protocol

# ----------------------------------------------------------------------------------------------- states
PROPOSED, APPROVED, QUEUED, SENT, DELIVERED, FAILED, ACKNOWLEDGED, ESCALATED = (
    "proposed", "approved", "queued", "sent", "delivered", "failed", "acknowledged", "escalated")

ALLOWED: dict[str, set[str]] = {
    PROPOSED: {APPROVED, FAILED},
    APPROVED: {QUEUED, FAILED},
    QUEUED: {SENT, DELIVERED, FAILED},
    SENT: {DELIVERED, FAILED, ACKNOWLEDGED},
    DELIVERED: {ACKNOWLEDGED},
    FAILED: {ESCALATED, QUEUED},          # QUEUED again = an explicit manual re-drive, never automatic
    ACKNOWLEDGED: set(),
    ESCALATED: {QUEUED, ACKNOWLEDGED},
}
IN_FLIGHT_OR_DONE = {QUEUED, SENT, DELIVERED, ACKNOWLEDGED, ESCALATED}   # a second dispatch must not send again


class IllegalTransition(Exception):
    pass


@dataclass
class NotificationRecord:
    idempotency_key: str
    action_id: str
    workflow_id: str
    patient_id: str
    channel: str
    action_type: str
    recipient_ref: str | None            # opaque user id; NEVER an address
    state: str = PROPOSED
    attempts: int = 0
    last_error_class: str | None = None
    provider_message_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def move(self, to: str, note: str = "") -> None:
        if to not in ALLOWED[self.state]:
            raise IllegalTransition(f"{self.state} -> {to}")
        self.history.append({"at": datetime.now(timezone.utc).isoformat(), "from": self.state, "to": to, "note": note})
        self.state = to


# ----------------------------------------------------------------------------------------------- provider API
@dataclass(frozen=True)
class SendRequest:
    idempotency_key: str
    recipient_ref: str
    subject: str
    body: str                           # generic, PHI-free text


@dataclass(frozen=True)
class ProviderResponse:
    outcome: str                        # "sent" | "delivered" | "rejected" | "transient_error"
    provider_message_id: str | None = None
    detail: str = ""                    # never logged verbatim


class NotificationProviderV2(ABC):
    channel = "abstract"

    @abstractmethod
    def send(self, request: SendRequest) -> ProviderResponse: ...

    def poll(self, provider_message_id: str) -> str | None:
        """Optional: 'delivered' / 'failed' / None (unknown). Providers without delivery receipts return None."""
        return None


class NullProviderV2(NotificationProviderV2):
    channel = "none"

    def send(self, request: SendRequest) -> ProviderResponse:
        return ProviderResponse("rejected", None, "no notification channel configured; nothing was sent")


class SmtpEmailProvider(NotificationProviderV2):
    """Plain SMTP. `resolve_address(recipient_ref)` maps the opaque user id to an address at send time, so addresses
    never sit in the outbox. Accepting a message (250) yields outcome 'sent' - SMTP gives no delivery receipt."""
    channel = "smtp"

    def __init__(self, host: str, port: int, from_addr: str, resolve_address: Callable[[str], str | None], *,
                 username: str = "", password: str = "", starttls: bool = True, timeout: float = 10.0):
        self.host, self.port, self.from_addr, self.resolve = host, port, from_addr, resolve_address
        self.username, self.password, self.starttls, self.timeout = username, password, starttls, timeout

    def send(self, request: SendRequest) -> ProviderResponse:
        try:
            to = self.resolve(request.recipient_ref)
        except Exception:
            return ProviderResponse("transient_error", None, "address_lookup_failed")
        if not to:
            return ProviderResponse("rejected", None, "no_address_for_recipient")
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.from_addr, to, request.subject
        msg["Message-ID"] = f"<{request.idempotency_key.replace(':', '.')}@neuro-mas>"
        msg.set_content(request.body)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as s:
                if self.starttls:
                    s.starttls(context=ssl.create_default_context())
                if self.username:
                    s.login(self.username, self.password)
                refused = s.send_message(msg)
        except smtplib.SMTPRecipientsRefused:
            return ProviderResponse("rejected", None, "recipient_refused")
        except smtplib.SMTPResponseException as e:
            return ProviderResponse("rejected" if 500 <= e.smtp_code < 600 else "transient_error", None, f"smtp_{e.smtp_code}")
        except (smtplib.SMTPException, OSError):
            return ProviderResponse("transient_error", None, "smtp_unreachable")
        if refused:
            return ProviderResponse("rejected", None, "recipient_refused")
        return ProviderResponse("sent", msg["Message-ID"], "accepted_by_relay")


# ----------------------------------------------------------------------------------------------- store
class NotificationStore(Protocol):
    async def get_by_key(self, key: str) -> NotificationRecord | None: ...
    async def save(self, rec: NotificationRecord) -> None: ...


class InMemoryNotificationStore:
    """Process-local store: for tests and single-process demos only. NOT durable."""

    def __init__(self):
        self.records: dict[str, NotificationRecord] = {}

    async def get_by_key(self, key):
        return self.records.get(key)

    async def save(self, rec):
        self.records[rec.idempotency_key] = rec


# ----------------------------------------------------------------------------------------------- dispatcher
AuditSink = Callable[[dict], None]
EscalationHook = Callable[[NotificationRecord], Awaitable[None]]


def generic_message(action_type: str, workflow_id: str) -> tuple[str, str]:
    """PHI-free text: says only that something needs attention and where to look."""
    subject = "NEURO-MAS: urgent patient alert" if action_type == "emergency_escalation" else "NEURO-MAS: notification"
    body = (f"A NEURO-MAS workflow ({workflow_id}) produced an item that needs your attention.\n"
            "Open the NEURO-MAS dashboard to review it. This message contains no patient details.")
    return subject, body


class NotificationDispatcher:
    def __init__(self, provider: NotificationProviderV2, store: NotificationStore, *, audit: AuditSink | None = None,
                 escalate: EscalationHook | None = None, max_attempts: int = 3, attempt_timeout: float = 10.0,
                 backoff_base: float = 1.0, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.provider, self.store, self.audit, self.escalate = provider, store, audit or (lambda e: None), escalate
        self.max_attempts, self.attempt_timeout, self.backoff_base, self.sleep = max_attempts, attempt_timeout, backoff_base, sleep
        self._locks: dict[str, asyncio.Lock] = {}

    def _emit(self, rec: NotificationRecord, event: str, **extra: Any) -> None:
        # ids, states and error CLASSES only: no recipient address, no body, no provider text
        self.audit({"event": event, "notification_id": rec.id, "action_id": rec.action_id, "workflow_id": rec.workflow_id,
                    "channel": rec.channel, "state": rec.state, "attempts": rec.attempts, **extra})

    async def dispatch(self, rec_in: NotificationRecord, *, approved: bool = True) -> NotificationRecord:
        """Idempotently deliver one notification. `approved=False` keeps it at 'proposed' (human approval pending)."""
        key = rec_in.idempotency_key
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:                                   # concurrent duplicates serialise, then see the stored state
            existing = await self.store.get_by_key(key)
            if existing is not None and existing.state in IN_FLIGHT_OR_DONE:
                self._emit(existing, "notification_duplicate_suppressed")
                return existing
            rec = existing or rec_in
            if rec.state == PROPOSED and not approved:
                await self.store.save(rec)
                return rec
            if rec.state == PROPOSED:
                rec.move(APPROVED)
            if rec.state == FAILED:
                return rec                                  # permanent failure: re-drive only by an explicit operator action
            if not rec.recipient_ref:
                rec.last_error_class = "no_recipient"
                rec.move(FAILED, "no_recipient")
                await self.store.save(rec)
                self._emit(rec, "notification_failed", error_class="no_recipient")
                await self._escalate(rec)
                return rec
            rec.move(QUEUED)
            await self.store.save(rec)                      # durable BEFORE the first send attempt
            self._emit(rec, "notification_queued")
            subject, body = generic_message(rec.action_type, rec.workflow_id)
            req = SendRequest(key, rec.recipient_ref, subject, body)

            while rec.attempts < self.max_attempts:
                rec.attempts += 1
                try:
                    resp = await asyncio.wait_for(asyncio.to_thread(self.provider.send, req), timeout=self.attempt_timeout)
                except asyncio.TimeoutError:
                    resp = ProviderResponse("transient_error", None, "timeout")
                    rec.last_error_class = "TimeoutError"
                except Exception as exc:
                    resp = ProviderResponse("transient_error", None, "exception")
                    rec.last_error_class = type(exc).__name__
                if resp.outcome in ("sent", "delivered"):
                    rec.provider_message_id = resp.provider_message_id
                    rec.last_error_class = None
                    rec.move(DELIVERED if resp.outcome == "delivered" else SENT)
                    await self.store.save(rec)
                    self._emit(rec, "notification_" + rec.state)
                    return rec
                if resp.outcome == "rejected":
                    rec.last_error_class = "rejected"
                    rec.move(FAILED, "rejected")
                    await self.store.save(rec)
                    self._emit(rec, "notification_failed", error_class="rejected")
                    await self._escalate(rec)
                    return rec
                rec.last_error_class = rec.last_error_class or "transient_error"
                await self.store.save(rec)
                self._emit(rec, "notification_attempt_failed", error_class=rec.last_error_class)
                if rec.attempts < self.max_attempts:
                    await self.sleep(self.backoff_base * (2 ** (rec.attempts - 1)))
            rec.move(FAILED, "retries_exhausted")
            await self.store.save(rec)
            self._emit(rec, "notification_failed", error_class=rec.last_error_class or "unknown")
            await self._escalate(rec)
            return rec

    async def _escalate(self, rec: NotificationRecord) -> None:
        """A delivery that failed for good must not disappear silently."""
        if self.escalate is None:
            self._emit(rec, "notification_escalation_unavailable")
            return
        try:
            await self.escalate(rec)
            rec.move(ESCALATED, "escalation_hook_ok")
            await self.store.save(rec)
            self._emit(rec, "notification_escalated")
        except Exception as exc:                            # the escalation path itself failed: stay FAILED, loudly
            self._emit(rec, "notification_escalation_failed", error_class=type(exc).__name__)

    async def refresh_delivery(self, rec: NotificationRecord) -> NotificationRecord:
        """Poll a provider that supports receipts: sent -> delivered / failed. Unknown stays 'sent'."""
        if rec.state != SENT or not rec.provider_message_id:
            return rec
        try:
            status = await asyncio.wait_for(asyncio.to_thread(self.provider.poll, rec.provider_message_id), timeout=self.attempt_timeout)
        except Exception:
            return rec
        if status == "delivered":
            rec.move(DELIVERED, "receipt")
        elif status == "failed":
            rec.move(FAILED, "receipt_failed")
            await self._escalate(rec)
        await self.store.save(rec)
        return rec


def action_status_for(rec: NotificationRecord) -> str:
    """Truthful action status from the notification state ('executed' only for confirmed delivery)."""
    return {DELIVERED: "executed", ACKNOWLEDGED: "executed", SENT: "sent_unconfirmed",
            FAILED: "delivery_failed", ESCALATED: "delivery_failed", QUEUED: "queued"}.get(rec.state, "proposed")


def recipient_for(action_type: str, patient: dict) -> str | None:
    """Opaque user id to notify. Clinician-facing actions go to the treating clinician, patient-facing ones to the
    patient's own account. Caregivers have no link in the schema, so caregiver notifications return None (=> failed
    with 'no_recipient', never silently dropped)."""
    if action_type in {"emergency_escalation", "clinician_notification", "follow_up_scheduling_suggestion"}:
        return patient.get("clinician_id")
    if action_type in {"reminder", "check_in_prompt", "symptom_questionnaire", "educational_intervention", "monitoring_request"}:
        return patient.get("user_id")
    return None


# Action types a dispatcher may deliver autonomously (mirrors notifier.NOTIFYING_ACTION_TYPES).
NOTIFYING_ACTION_TYPES_FOR_DISPATCH = {
    "emergency_escalation", "clinician_notification", "reminder", "check_in_prompt", "symptom_questionnaire",
    "educational_intervention", "follow_up_scheduling_suggestion", "monitoring_request", "caregiver_notification",
}
