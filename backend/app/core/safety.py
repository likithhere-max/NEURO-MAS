"""
Safety Policy Engine (spec section 12) + Autonomy Hierarchy (spec section 3).

This module is the single authority for "is this action allowed to happen
autonomously." No agent, and no LLM output, can bypass it. It is pure,
deterministic Python -- easy to unit-test exhaustively (see
backend/tests/test_safety.py), which is the whole point of keeping it out of
model-dependent code paths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

from app.core.config import get_settings


class AutonomyLevel(IntEnum):
    OBSERVE = 0
    ANALYZE = 1
    RECOMMEND = 2
    EXECUTE_LOW_RISK = 3
    ESCALATE = 4
    HUMAN_AUTHORIZATION_REQUIRED = 5


# Action catalog: every action type NEURO-MAS can propose, tagged with the
# autonomy level it belongs to and whether it may ever run without a human
# clicking "approve" first.
ACTION_CATALOG: dict[str, dict] = {
    "reminder": {"level": AutonomyLevel.EXECUTE_LOW_RISK, "auto_executable": True},
    "check_in_prompt": {"level": AutonomyLevel.EXECUTE_LOW_RISK, "auto_executable": True},
    "symptom_questionnaire": {"level": AutonomyLevel.EXECUTE_LOW_RISK, "auto_executable": True},
    "educational_intervention": {"level": AutonomyLevel.EXECUTE_LOW_RISK, "auto_executable": True},
    "monitoring_request": {"level": AutonomyLevel.EXECUTE_LOW_RISK, "auto_executable": True},
    "caregiver_notification": {"level": AutonomyLevel.EXECUTE_LOW_RISK, "auto_executable": True},
    "clinician_notification": {"level": AutonomyLevel.RECOMMEND, "auto_executable": True},
    "follow_up_scheduling_suggestion": {"level": AutonomyLevel.RECOMMEND, "auto_executable": True},
    "emergency_escalation": {"level": AutonomyLevel.ESCALATE, "auto_executable": True},
    "prescription_modification": {"level": AutonomyLevel.HUMAN_AUTHORIZATION_REQUIRED, "auto_executable": False},
    "medication_initiation": {"level": AutonomyLevel.HUMAN_AUTHORIZATION_REQUIRED, "auto_executable": False},
    "diagnosis_confirmation": {"level": AutonomyLevel.HUMAN_AUTHORIZATION_REQUIRED, "auto_executable": False},
    "invasive_intervention": {"level": AutonomyLevel.HUMAN_AUTHORIZATION_REQUIRED, "auto_executable": False},
}

RESTRICTED_ACTION_TYPES = {
    k for k, v in ACTION_CATALOG.items() if v["level"] == AutonomyLevel.HUMAN_AUTHORIZATION_REQUIRED
}


def _is_finite_number(x) -> bool:
    """True only for a real, finite int/float (bool, None, NaN, +-inf and non-numbers are all rejected)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass
class SafetyDecision:
    permitted_autonomously: bool
    autonomy_level: AutonomyLevel
    requires_human_approval: bool
    reasons: list[str]


def evaluate_action(
    action_type: str,
    *,
    is_emergency: bool,
    disagreement: float,
    top_confidence: float,
    data_quality: float,
    medication_related: bool = False,
    clinician_restricted_actions: list[str] | None = None,
) -> SafetyDecision:
    """The single choke point every proposed action must pass through.

    Deterministic rule evaluation only -- see module docstring.
    """
    settings = get_settings()
    reasons: list[str] = []
    clinician_restricted_actions = clinician_restricted_actions or []

    if action_type not in ACTION_CATALOG:
        return SafetyDecision(
            permitted_autonomously=False,
            autonomy_level=AutonomyLevel.HUMAN_AUTHORIZATION_REQUIRED,
            requires_human_approval=True,
            reasons=[f"Unknown action_type '{action_type}' -- defaulting to human authorization required."],
        )

    spec = ACTION_CATALOG[action_type]
    level = spec["level"]

    # RULE 1: restricted action types (medication changes, diagnosis
    # confirmation, invasive intervention) ALWAYS require human authorization.
    # No confidence score, consensus level, or emergency flag overrides this.
    if action_type in RESTRICTED_ACTION_TYPES or medication_related and action_type == "prescription_modification":
        reasons.append("Action type is in the restricted catalog (medication/diagnosis/invasive) -- "
                        "human clinician authorization is always required, by policy, not by confidence.")
        return SafetyDecision(False, level, True, reasons)  # never auto-executable

    # RULE 2: clinician-defined per-patient restriction overrides everything else.
    if action_type in clinician_restricted_actions:
        reasons.append(f"Clinician has explicitly restricted '{action_type}' for this patient.")
        return SafetyDecision(False, level, True, reasons)

    # RULE 3: the dedicated emergency-escalation action type always fires
    # immediately, regardless of consensus/confidence (a confused set of
    # agents should never block a genuine emergency). This is scoped to
    # action_type == "emergency_escalation" specifically -- an active
    # emergency must NOT silently reclassify unrelated low-risk actions
    # (e.g. a medication reminder) as escalations; those still go through
    # their own RULE 4/5 checks below.
    if level == AutonomyLevel.ESCALATE:
        reasons.append("Escalation-type action: executes immediately, independent of consensus/disagreement "
                        "state, per policy for time-critical events." +
                        (" Triggered by the deterministic emergency detector." if is_emergency else ""))
        return SafetyDecision(True, level, False, reasons)

    # RULE 4: low-risk actions still need minimum confidence + data quality +
    # bounded disagreement before running autonomously.
    if level == AutonomyLevel.EXECUTE_LOW_RISK:
        # Phase 3 (D01): every comparison below is False for NaN, so a NaN/None/inf gate input used to FAIL OPEN and let the
        # action run autonomously. A gate input that is not a finite number is missing information: fail closed.
        bad = [n for n, v in (("disagreement", disagreement), ("top_confidence", top_confidence),
                              ("data_quality", data_quality)) if not _is_finite_number(v)]
        if bad:
            reasons.append(f"Gate input(s) {bad} are not a finite number (non-finite or missing); cannot verify the "
                           f"autonomy preconditions, so human approval is required.")
            return SafetyDecision(False, level, True, reasons)
        if disagreement > settings.consensus_disagreement_threshold:
            reasons.append(f"Disagreement {disagreement} exceeds threshold "
                            f"{settings.consensus_disagreement_threshold}; requesting additional analysis "
                            f"instead of acting autonomously.")
            return SafetyDecision(False, level, True, reasons)
        if top_confidence < settings.min_confidence_for_autonomous_action:
            reasons.append(f"Top hypothesis confidence {top_confidence} below minimum "
                            f"{settings.min_confidence_for_autonomous_action} for autonomous action.")
            return SafetyDecision(False, level, True, reasons)
        if data_quality < 0.3:
            reasons.append(f"Data quality {data_quality} too low to act autonomously; requesting more data.")
            return SafetyDecision(False, level, True, reasons)
        reasons.append("Confidence, data quality and disagreement all within bounds for a low-risk action.")
        return SafetyDecision(True, level, False, reasons)

    # RULE 5: RECOMMEND-level actions (notifications) are allowed to fire
    # more liberally since they only inform a human, they don't act on the
    # patient directly.
    if level == AutonomyLevel.RECOMMEND:
        reasons.append("Notification-only action; permitted to proceed to inform a human.")
        return SafetyDecision(True, level, False, reasons)

    reasons.append("Fell through to default: human authorization required.")
    return SafetyDecision(False, level, True, reasons)
