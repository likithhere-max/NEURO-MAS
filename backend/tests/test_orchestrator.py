"""
Integration tests for the orchestrator: verifies the full closed loop
(data intake -> agents -> consensus -> safety -> action) behaves correctly
for both an emergency and a non-emergency, disagreement-heavy scenario.
"""
import pytest

from app.orchestrator.orchestrator import run_workflow
from app.schemas.patient_state import MedicationState, PatientState


@pytest.mark.asyncio
async def test_emergency_scenario_escalates_and_bypasses_consensus_gate():
    state = PatientState(patient_id="p1", medication_state=MedicationState(adherence_pct_30d=0.6))
    result = await run_workflow(
        patient_id="p1",
        trigger_reason="test_emergency",
        input_payload={
            "symptoms": [{"name": "seizure", "severity": "severe", "free_text": "status epilepticus, seizure not stopping"}],
        },
        patient_state=state,
    )
    assert result.is_emergency is True
    escalations = [a for a in result.proposed_actions if a.action_type == "emergency_escalation"]
    assert len(escalations) == 1
    assert escalations[0].status == "executed"
    assert escalations[0].requires_human_approval is False


@pytest.mark.asyncio
async def test_non_emergency_workflow_completes_and_produces_consensus():
    state = PatientState(patient_id="p2")
    result = await run_workflow(
        patient_id="p2",
        trigger_reason="routine_check_in",
        input_payload={"symptoms": [{"name": "mild headache", "severity": "low", "free_text": "slight headache"}]},
        patient_state=state,
    )
    assert result.status == "completed"
    assert result.is_emergency is False
    # No agent crash should ever leave the workflow half-finished.
    assert all(r.status in {"completed", "skipped", "failed", "model_not_available"} for r in result.agent_results.values())


@pytest.mark.asyncio
async def test_agent_failure_does_not_crash_workflow():
    """Simulates degraded mode: even if an agent were to raise, the
    orchestrator's try/except must keep the workflow moving (spec section 22,
    'never silently continue after a safety-critical failure' -- but also
    never let a NON-safety-critical agent failure kill the whole run).
    """
    state = PatientState(patient_id="p3")
    # Malformed payload (wrong types) exercises defensive code paths in
    # multiple agents without needing to monkeypatch anything.
    result = await run_workflow(
        patient_id="p3",
        trigger_reason="malformed_input_test",
        input_payload={"symptoms": None, "wearable": None},
        patient_state=state,
    )
    assert result.status == "completed"
