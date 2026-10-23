"""Notification outbox, state machine, idempotency, retries, escalation (Phase 2, G).
All sends go to a MOCK provider (tests/helpers_notifications.py): this verifies OUR dispatch logic. It does not
demonstrate that any real e-mail/SMS/page was ever delivered."""
import asyncio
import json
import smtplib

import pytest

from app.services import notification_dispatch as nd
from app.services.notification_dispatch import (
    ALLOWED, IllegalTransition, InMemoryNotificationStore, NotificationDispatcher, NotificationRecord, SmtpEmailProvider,
)
from tests.helpers_notifications import MockProvider

ADDR = "secret.person@example.org"


def rec(action="a1", rcpt="user-1", typ="emergency_escalation", channel="mock"):
    return NotificationRecord(idempotency_key=f"{action}:{channel}", action_id=action, workflow_id="w1", patient_id="p1",
                              channel=channel, action_type=typ, recipient_ref=rcpt)


class Harness:
    def __init__(self, provider, **kw):
        self.events, self.sleeps, self.escalated = [], [], []
        self.store = InMemoryNotificationStore()

        async def sleep(d): self.sleeps.append(d)

        async def esc(r): self.escalated.append(r.action_id)
        self.d = NotificationDispatcher(provider, self.store, audit=self.events.append, escalate=kw.pop("escalate", esc),
                                        sleep=sleep, **{"max_attempts": 3, "attempt_timeout": 1.0, "backoff_base": 1.0, **kw})


# ---------------------------------------------------------------- happy paths & truthfulness
@pytest.mark.asyncio
async def test_accepted_by_provider_is_sent_not_executed():
    h = Harness(MockProvider([("sent", "")]))
    r = await h.d.dispatch(rec())
    assert r.state == "sent" and nd.action_status_for(r) == "sent_unconfirmed"


@pytest.mark.asyncio
async def test_only_confirmed_delivery_is_executed():
    h = Harness(MockProvider([("delivered", "")]))
    r = await h.d.dispatch(rec())
    assert r.state == "delivered" and nd.action_status_for(r) == "executed"


@pytest.mark.asyncio
async def test_receipt_poll_promotes_sent_to_delivered():
    p = MockProvider([("sent", "")])
    h = Harness(p)
    r = await h.d.dispatch(rec())
    assert (await h.d.refresh_delivery(r)).state == "sent"            # unknown receipt: stays sent
    p.receipts[r.provider_message_id] = "delivered"
    assert (await h.d.refresh_delivery(r)).state == "delivered"


@pytest.mark.asyncio
async def test_negative_receipt_fails_and_escalates():
    p = MockProvider([("sent", "")])
    h = Harness(p)
    r = await h.d.dispatch(rec())
    p.receipts[r.provider_message_id] = "failed"
    r = await h.d.refresh_delivery(r)
    assert r.state == "escalated" and h.escalated == ["a1"]


# ---------------------------------------------------------------- retries
@pytest.mark.asyncio
async def test_transient_failures_are_retried_with_backoff_then_succeed():
    p = MockProvider([("transient_error", "x"), ("transient_error", "x"), ("sent", "")])
    h = Harness(p)
    r = await h.d.dispatch(rec())
    assert r.state == "sent" and r.attempts == 3 and len(p.calls) == 3
    assert h.sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_retries_are_bounded_and_failure_escalates():
    p = MockProvider([("transient_error", "x")])
    h = Harness(p, max_attempts=3)
    r = await h.d.dispatch(rec())
    assert len(p.calls) == 3 and r.attempts == 3 and r.state == "escalated" and h.escalated == ["a1"]
    assert nd.action_status_for(r) == "delivery_failed"


@pytest.mark.asyncio
async def test_permanent_rejection_is_not_retried():
    p = MockProvider([("rejected", "bad")])
    h = Harness(p)
    r = await h.d.dispatch(rec())
    assert len(p.calls) == 1 and r.state == "escalated" and r.last_error_class == "rejected"


@pytest.mark.asyncio
async def test_provider_exception_counts_as_outage_and_is_retried():
    p = MockProvider([ConnectionError("down"), ConnectionError("down"), ("sent", "")])
    h = Harness(p)
    r = await h.d.dispatch(rec())
    assert r.state == "sent" and len(p.calls) == 3


@pytest.mark.asyncio
async def test_hung_provider_times_out_per_attempt():
    p = MockProvider([("sent", "")], delay=0.4)
    h = Harness(p, attempt_timeout=0.05, max_attempts=2)
    r = await h.d.dispatch(rec())
    assert r.state == "escalated" and r.last_error_class == "TimeoutError" and r.attempts == 2


# ---------------------------------------------------------------- idempotency
@pytest.mark.asyncio
async def test_duplicate_dispatch_sends_exactly_once():
    p = MockProvider([("sent", "")])
    h = Harness(p)
    first = await h.d.dispatch(rec())
    again = await h.d.dispatch(rec())
    assert len(p.calls) == 1 and again.state == first.state
    assert any(e["event"] == "notification_duplicate_suppressed" for e in h.events)


@pytest.mark.asyncio
async def test_concurrent_duplicates_send_exactly_once():
    p = MockProvider([("sent", "")], delay=0.05)
    h = Harness(p)
    res = await asyncio.gather(*[h.d.dispatch(rec()) for _ in range(10)])
    assert len(p.calls) == 1 and {r.state for r in res} == {"sent"}


@pytest.mark.asyncio
async def test_failed_notification_is_not_silently_resent():
    p = MockProvider([("rejected", "")])
    h = Harness(p, escalate=None)             # no hook: stays FAILED
    r = await h.d.dispatch(rec())
    assert r.state == "failed"
    p.script = [("sent", "")]
    again = await h.d.dispatch(rec())
    assert len(p.calls) == 1 and again.state == "failed"


@pytest.mark.asyncio
async def test_distinct_actions_and_channels_do_not_collide():
    p = MockProvider([("sent", "")])
    h = Harness(p)
    await h.d.dispatch(rec("a1")); await h.d.dispatch(rec("a2")); await h.d.dispatch(rec("a1", channel="smtp"))
    assert len(p.calls) == 3


# ---------------------------------------------------------------- approval, recipients, escalation failure
@pytest.mark.asyncio
async def test_unapproved_notification_is_not_sent_until_approved():
    p = MockProvider([("sent", "")])
    h = Harness(p)
    r = await h.d.dispatch(rec(), approved=False)
    assert r.state == "proposed" and p.calls == []
    r = await h.d.dispatch(rec(), approved=True)
    assert r.state == "sent" and len(p.calls) == 1


@pytest.mark.asyncio
async def test_missing_recipient_fails_without_sending_and_escalates():
    p = MockProvider([("sent", "")])
    h = Harness(p)
    r = await h.d.dispatch(rec(rcpt=None, typ="caregiver_notification"))
    assert p.calls == [] and r.state == "escalated" and r.last_error_class == "no_recipient"


@pytest.mark.asyncio
async def test_escalation_failure_leaves_record_failed_and_is_audited():
    async def broken(r): raise RuntimeError("db down")
    h = Harness(MockProvider([("rejected", "")]), escalate=broken)
    r = await h.d.dispatch(rec())
    assert r.state == "failed"
    assert any(e["event"] == "notification_escalation_failed" for e in h.events)


# ---------------------------------------------------------------- state machine
def test_illegal_transitions_are_rejected():
    r = rec()
    for bad in ("sent", "delivered", "acknowledged", "escalated"):
        with pytest.raises(IllegalTransition):
            r.move(bad)
    r.move("approved"); r.move("queued"); r.move("sent"); r.move("delivered"); r.move("acknowledged")
    for to in ("queued", "sent", "failed"):
        with pytest.raises(IllegalTransition):
            r.move(to)


def test_every_state_is_defined_and_terminal_states_have_no_exit():
    assert set(ALLOWED) == {"proposed", "approved", "queued", "sent", "delivered", "failed", "acknowledged", "escalated"}
    assert ALLOWED["acknowledged"] == set()


@pytest.mark.asyncio
async def test_history_records_every_transition():
    h = Harness(MockProvider([("sent", "")]))
    r = await h.d.dispatch(rec())
    assert [x["to"] for x in r.history] == ["approved", "queued", "sent"]


# ---------------------------------------------------------------- redaction of audit trail & message body
@pytest.mark.asyncio
async def test_audit_events_and_message_hold_no_address_body_or_provider_text():
    p = MockProvider([("transient_error", f"upstream said {ADDR}"), ("sent", "")])
    h = Harness(p)
    r = await h.d.dispatch(rec())
    blob = json.dumps(h.events) + json.dumps(r.history)
    assert ADDR not in blob and "upstream said" not in blob and "user-1" not in blob        # no address / detail / recipient id
    sent_text = p.calls[0].subject + p.calls[0].body
    assert "p1" not in sent_text                                   # patient id is not in the message
    assert "no patient details" in p.calls[0].body


def test_generic_message_has_no_clinical_content():
    s, b = nd.generic_message("emergency_escalation", "wf-9")
    assert "wf-9" in b and "urgent" in s.lower()
    for word in ("seizure", "stroke", "tremor", "diagnos", "medication"):
        assert word not in (s + b).lower()


def test_recipient_mapping_is_explicit_and_caregiver_has_none():
    pt = {"clinician_id": "doc", "user_id": "pat"}
    assert nd.recipient_for("emergency_escalation", pt) == "doc"
    assert nd.recipient_for("reminder", pt) == "pat"
    assert nd.recipient_for("caregiver_notification", pt) is None


# ---------------------------------------------------------------- SMTP provider against a stand-in for smtplib (MOCK)
class FakeSMTP:
    instances, behaviour = [], {}

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.tls, self.login_args, self.sent = host, port, False, None, []
        FakeSMTP.instances.append(self)
        if FakeSMTP.behaviour.get("connect"):
            raise FakeSMTP.behaviour["connect"]

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def starttls(self, context=None): self.tls = True
    def login(self, u, p): self.login_args = (u, p)

    def send_message(self, msg):
        if FakeSMTP.behaviour.get("send"):
            raise FakeSMTP.behaviour["send"]
        self.sent.append(msg)
        return FakeSMTP.behaviour.get("refused", {})


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances, FakeSMTP.behaviour = [], {}
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


def prov(resolve=lambda u: ADDR, **kw):
    return SmtpEmailProvider("smtp.test", 587, "noreply@test", resolve, username="u", password="p", **kw)


REQ = nd.SendRequest("a1:smtp", "user-1", "S", "B")


def test_smtp_accept_is_sent_with_tls_login_and_message_id(smtp):
    r = prov().send(REQ)
    i = smtp.instances[0]
    assert r.outcome == "sent" and i.tls and i.login_args == ("u", "p")
    assert i.sent[0]["To"] == ADDR and "a1.smtp@neuro-mas" in i.sent[0]["Message-ID"]


@pytest.mark.parametrize("exc,expected", [
    (smtplib.SMTPRecipientsRefused({ADDR: (550, b"no")}), "rejected"),
    (smtplib.SMTPResponseException(550, b"mailbox unavailable"), "rejected"),
    (smtplib.SMTPResponseException(421, b"try later"), "transient_error"),
    (smtplib.SMTPServerDisconnected("gone"), "transient_error"),
    (OSError("network unreachable"), "transient_error"),
])
def test_smtp_errors_are_classified_permanent_vs_transient(smtp, exc, expected):
    smtp.behaviour["send"] = exc
    assert prov().send(REQ).outcome == expected


def test_smtp_connect_failure_is_transient(smtp):
    smtp.behaviour["connect"] = ConnectionRefusedError()
    assert prov().send(REQ).outcome == "transient_error"


def test_smtp_partial_refusal_is_rejected(smtp):
    smtp.behaviour["refused"] = {ADDR: (550, b"x")}
    assert prov().send(REQ).outcome == "rejected"


def test_smtp_unknown_or_unresolvable_address(smtp):
    assert prov(lambda u: None).send(REQ).outcome == "rejected"
    def boom(u): raise RuntimeError("auth api down")
    assert prov(boom).send(REQ).outcome == "transient_error"
    assert smtp.instances == []               # no connection attempted without an address


def test_smtp_response_detail_never_contains_the_address(smtp):
    smtp.behaviour["send"] = smtplib.SMTPResponseException(550, f"no such user {ADDR}".encode())
    assert ADDR not in prov().send(REQ).detail


# ---------------------------------------------------------------- configuration safety
def test_smtp_channel_requires_host_and_sender():
    from app.core.config import Settings
    with pytest.raises(Exception, match="SMTP_HOST"):
        Settings(_env_file=None, notification_channel="smtp")
    Settings(_env_file=None, notification_channel="smtp", smtp_host="h", smtp_from="f@x.org")


def test_default_channel_sends_nothing():
    from app.core.config import Settings
    from app.services.notification_service import build_dispatcher
    assert Settings(_env_file=None).notification_channel == "none"
    assert build_dispatcher(object()) is None


def test_null_provider_is_truthful():
    assert nd.NullProviderV2().send(REQ).outcome == "rejected"
