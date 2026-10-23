"""Wiring of the notification dispatcher into the application (Phase 2, G). Nothing is sent unless the owner sets
NOTIFICATION_CHANNEL (and supplies that channel's credentials); the default is 'none' and actions stay 'proposed'."""
from __future__ import annotations

import logging
from dataclasses import asdict

from app.core.config import get_settings
from app.services.notification_dispatch import (
    NOTIFYING_ACTION_TYPES_FOR_DISPATCH,
    NotificationDispatcher,
    NotificationRecord,
    SmtpEmailProvider,
    action_status_for,
    recipient_for,
)

logger = logging.getLogger("neuromas.notifications")


class SupabaseNotificationStore:
    """Durable outbox in the `notifications` table (see scripts/migration_phase2_notifications.sql). Uses the
    service-role client: the outbox is server-owned and the table has no end-user policies."""

    TABLE = "notifications"

    def __init__(self, admin_client):
        self.c = admin_client

    async def get_by_key(self, key):
        import asyncio
        rows = await asyncio.to_thread(lambda: self.c.table(self.TABLE).select("*").eq("idempotency_key", key).execute().data)
        return NotificationRecord(**{k: rows[0][k] for k in NotificationRecord.__dataclass_fields__ if k in rows[0]}) if rows else None

    async def save(self, rec):
        import asyncio
        row = asdict(rec)
        await asyncio.to_thread(lambda: self.c.table(self.TABLE).upsert(row, on_conflict="idempotency_key").execute())


def build_dispatcher(admin_client, escalate=None, audit=None) -> NotificationDispatcher | None:
    s = get_settings()
    channel = (s.notification_channel or "none").lower()
    if channel == "none":
        return None
    if channel != "smtp":
        logger.error("unknown NOTIFICATION_CHANNEL=%r: no notifications will be sent", channel)
        return None

    def resolve(user_id: str):
        return admin_client.auth.admin.get_user_by_id(user_id).user.email

    provider = SmtpEmailProvider(s.smtp_host, s.smtp_port, s.smtp_from, resolve, username=s.smtp_username,
                                 password=s.smtp_password, starttls=s.smtp_starttls, timeout=s.notification_attempt_timeout_s)
    return NotificationDispatcher(provider, SupabaseNotificationStore(admin_client), audit=audit, escalate=escalate,
                                  max_attempts=s.notification_max_attempts, attempt_timeout=s.notification_attempt_timeout_s,
                                  backoff_base=s.notification_backoff_base_s)


async def dispatch_workflow_notifications(state, patient: dict, dispatcher: NotificationDispatcher, client) -> dict[str, str]:
    """Deliver the autonomous (no-approval-needed) notifying actions of a stored workflow. Returns {action_id: status}
    and updates each action's stored status truthfully. Never raises into the request path."""
    import asyncio
    channel = (get_settings().notification_channel or "none").lower()
    out: dict[str, str] = {}
    for a in state.proposed_actions:
        if a.requires_human_approval or a.action_type not in NOTIFYING_ACTION_TYPES_FOR_DISPATCH or a.status != "proposed":
            continue
        rec = NotificationRecord(idempotency_key=f"{a.action_id}:{channel}", action_id=a.action_id, workflow_id=a.workflow_id,
                                 patient_id=a.patient_id, channel=channel, action_type=a.action_type,
                                 recipient_ref=recipient_for(a.action_type, patient))
        try:
            done = await dispatcher.dispatch(rec)
            status = action_status_for(done)
            out[a.action_id] = status
            a.status = status
            await asyncio.to_thread(lambda: client.table("actions").update({"status": status}).eq("id", a.action_id).execute())
        except Exception as exc:                      # a notification fault must not break the stored assessment
            logger.error("notification_dispatch_crashed action_id=%s error_class=%s", a.action_id, type(exc).__name__)
            out[a.action_id] = "delivery_failed"
    return out


def make_alert_escalation(client):
    """Escalation hook: a delivery that failed for good becomes a critical, open alert that clinicians can see."""
    async def hook(rec: NotificationRecord) -> None:
        import asyncio
        row = {"patient_id": rec.patient_id, "workflow_id": rec.workflow_id, "severity": "critical", "status": "open",
               "reason": f"NOTIFICATION DELIVERY FAILED for action {rec.action_id} ({rec.action_type}) after {rec.attempts} "
                         f"attempt(s); nobody has been told. Review the workflow manually.",
               "evidence_json": [f"notification:{rec.id}", f"error_class:{rec.last_error_class}"]}
        await asyncio.to_thread(lambda: client.table("alerts").insert(row).execute())
    return hook
