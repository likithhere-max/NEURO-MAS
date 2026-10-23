"""
Notifier: the single seam between "an action was approved by the safety
engine" and "someone was actually told".

Audit finding (Critical): actions.build_action set status="executed" for
permitted actions -- including emergency_escalation -- although NO code anywhere
sends an email/SMS/push/page. A UI badge saying "executed" therefore claimed a
delivery that never happened.

This module makes delivery explicit. An action may only be marked "executed"
if a configured provider returns delivered=True. The default provider is
NullNotificationProvider, which truthfully reports delivered=False. No real
channel (SMTP/SMS/push/pager) is implemented in this repository; wiring one is
an outstanding item (see docs/audit/OUTSTANDING_ISSUES.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DeliveryResult:
    delivered: bool
    channel: str
    detail: str


class NotificationProvider(Protocol):
    def deliver(self, *, action_type: str, patient_id: str, workflow_id: str, message: str) -> DeliveryResult: ...


class NullNotificationProvider:
    channel = "none"

    def deliver(self, *, action_type: str, patient_id: str, workflow_id: str, message: str) -> DeliveryResult:
        return DeliveryResult(False, self.channel, "no notification channel configured; nothing was sent")


_provider: NotificationProvider = NullNotificationProvider()


def get_provider() -> NotificationProvider:
    return _provider


def set_provider(provider: NotificationProvider) -> None:
    """Install a provider (production wiring, or a fake in tests)."""
    global _provider
    _provider = provider


# Action types whose "execution" is a notification/delivery to a person.
NOTIFYING_ACTION_TYPES = {
    "emergency_escalation", "clinician_notification", "caregiver_notification", "reminder",
    "check_in_prompt", "symptom_questionnaire", "educational_intervention", "follow_up_scheduling_suggestion",
    "monitoring_request",
}
