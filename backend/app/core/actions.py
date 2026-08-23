"""
Autonomous Action Engine (spec section 13).

Turns a personalization plan item + safety decision into a persisted,
auditable Action record. This module never decides IF an action is allowed
(that's app.core.safety) -- it only shapes and, for auto-executable actions,
"executes" them (in this prototype, execution means recording the action and
emitting a notification-shaped payload; see docs/SIMULATED_VS_REAL.md for
what a real notification/paging integration would replace this with).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.safety import SafetyDecision, evaluate_action

# Maps a free-text personalization plan item to a concrete action_type in
# the safety catalog. A real system would have the personalization agent
# emit typed action requests directly; this mapping keeps the prototype's
# personalization agent free-text-friendly while still routing through the
# typed safety catalog.
PLAN_ITEM_ACTION_TYPE_HINTS: list[tuple[str, str]] = [
    ("reminder", "reminder"),
    ("check-in", "check_in_prompt"),
    ("questionnaire", "symptom_questionnaire"),
    ("education", "educational_intervention"),
    ("monitoring", "monitoring_request"),
    ("caregiver", "caregiver_notification"),
    ("clinician", "clinician_notification"),
    ("follow-up", "follow_up_scheduling_suggestion"),
    ("follow up", "follow_up_scheduling_suggestion"),
]


def infer_action_type(plan_item_text: str) -> str:
    lowered = plan_item_text.lower()
    for keyword, action_type in PLAN_ITEM_ACTION_TYPE_HINTS:
        if keyword in lowered:
            return action_type
    return "monitoring_request"


@dataclass
class ProposedAction:
    action_id: str
    patient_id: str
    workflow_id: str
    action_type: str
    reason: str
    evidence: list[str]
    risk_level: str
    autonomy_level: int
    requires_human_approval: bool
    approved_by: str | None
    status: str  # "proposed" | "executed" | "pending_approval" | "rejected"
    created_at: str
    safety_reasons: list[str] = field(default_factory=list)


RISK_LEVEL_BY_AUTONOMY = {0: "none", 1: "none", 2: "low", 3: "low", 4: "high", 5: "restricted"}


def build_action(
    *,
    patient_id: str,
    workflow_id: str,
    plan_item: str,
    reason: str,
    evidence: list[str],
    is_emergency: bool,
    disagreement: float,
    top_confidence: float,
    data_quality: float,
    clinician_restricted_actions: list[str],
    action_type: str | None = None,
) -> ProposedAction:
    # action_type is only forced to "emergency_escalation" when the caller
    # explicitly asks for it (the orchestrator's dedicated emergency action).
    # An active emergency must NOT silently relabel unrelated personalization
    # plan items (reminders, monitoring requests) as escalations -- each
    # keeps its own inferred type and goes through its own safety check.
    if action_type is None:
        action_type = infer_action_type(plan_item)
    decision: SafetyDecision = evaluate_action(
        action_type,
        is_emergency=is_emergency,
        disagreement=disagreement,
        top_confidence=top_confidence,
        data_quality=data_quality,
        clinician_restricted_actions=clinician_restricted_actions,
    )

    status = "proposed"
    if decision.requires_human_approval:
        status = "pending_approval"
    elif decision.permitted_autonomously:
        status = "executed"  # low-risk/notification actions execute immediately in this prototype

    return ProposedAction(
        action_id=str(uuid.uuid4()),
        patient_id=patient_id,
        workflow_id=workflow_id,
        action_type=action_type,
        reason=reason,
        evidence=evidence,
        risk_level=RISK_LEVEL_BY_AUTONOMY[int(decision.autonomy_level)],
        autonomy_level=int(decision.autonomy_level),
        requires_human_approval=decision.requires_human_approval,
        approved_by=None,
        status=status,
        created_at=datetime.now(timezone.utc).isoformat(),
        safety_reasons=decision.reasons,
    )
