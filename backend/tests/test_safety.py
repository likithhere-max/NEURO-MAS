"""
Safety-bypass tests (spec section 26): the whole point of the safety engine
is that no confidence score, consensus result, or model output can grant
permission to do something restricted. These tests exhaustively probe that
boundary.
"""
from app.core.safety import AutonomyLevel, evaluate_action


def test_prescription_modification_never_auto_executable_even_with_perfect_confidence():
    decision = evaluate_action(
        "prescription_modification",
        is_emergency=False,
        disagreement=0.0,
        top_confidence=1.0,
        data_quality=1.0,
    )
    assert decision.permitted_autonomously is False
    assert decision.requires_human_approval is True
    assert decision.autonomy_level == AutonomyLevel.HUMAN_AUTHORIZATION_REQUIRED


def test_prescription_modification_not_bypassed_by_emergency_flag():
    decision = evaluate_action(
        "prescription_modification",
        is_emergency=True,
        disagreement=0.0,
        top_confidence=1.0,
        data_quality=1.0,
    )
    assert decision.permitted_autonomously is False
    assert decision.requires_human_approval is True


def test_diagnosis_confirmation_always_requires_human_approval():
    decision = evaluate_action(
        "diagnosis_confirmation", is_emergency=False, disagreement=0.0, top_confidence=1.0, data_quality=1.0,
    )
    assert decision.requires_human_approval is True


def test_low_risk_action_blocked_by_high_disagreement():
    decision = evaluate_action(
        "reminder", is_emergency=False, disagreement=0.8, top_confidence=0.9, data_quality=0.9,
    )
    assert decision.permitted_autonomously is False
    assert decision.requires_human_approval is True


def test_low_risk_action_blocked_by_low_confidence():
    decision = evaluate_action(
        "reminder", is_emergency=False, disagreement=0.1, top_confidence=0.2, data_quality=0.9,
    )
    assert decision.permitted_autonomously is False


def test_low_risk_action_permitted_when_all_thresholds_met():
    decision = evaluate_action(
        "reminder", is_emergency=False, disagreement=0.1, top_confidence=0.9, data_quality=0.9,
    )
    assert decision.permitted_autonomously is True
    assert decision.requires_human_approval is False


def test_emergency_escalation_bypasses_disagreement_and_confidence():
    decision = evaluate_action(
        "emergency_escalation", is_emergency=True, disagreement=1.0, top_confidence=0.0, data_quality=0.0,
    )
    assert decision.permitted_autonomously is True
    assert decision.requires_human_approval is False
    assert decision.autonomy_level == AutonomyLevel.ESCALATE


def test_active_emergency_does_not_reclassify_unrelated_low_risk_action():
    """Regression test: an active emergency must not silently turn an
    unrelated reminder into an auto-executed 'escalation' -- it should still
    go through its own confidence/disagreement gate.
    """
    decision = evaluate_action(
        "reminder", is_emergency=True, disagreement=0.9, top_confidence=0.1, data_quality=0.9,
    )
    assert decision.autonomy_level == AutonomyLevel.EXECUTE_LOW_RISK
    assert decision.permitted_autonomously is False  # blocked by its own disagreement threshold, not auto-escalated


def test_clinician_restriction_overrides_otherwise_permitted_action():
    decision = evaluate_action(
        "reminder", is_emergency=False, disagreement=0.0, top_confidence=1.0, data_quality=1.0,
        clinician_restricted_actions=["reminder"],
    )
    assert decision.permitted_autonomously is False
    assert decision.requires_human_approval is True


def test_unknown_action_type_defaults_to_human_authorization():
    decision = evaluate_action(
        "do_something_undefined", is_emergency=False, disagreement=0.0, top_confidence=1.0, data_quality=1.0,
    )
    assert decision.requires_human_approval is True
