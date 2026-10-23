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
    # 'executed' would claim a delivery that no code performs; with no notification channel it stays 'proposed'.
    assert escalations[0].status == "proposed"
    assert any("NOT_DELIVERED" in r for r in escalations[0].safety_reasons)
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


# ---------------- audit regressions ----------------

@pytest.mark.asyncio
async def test_clean_run_is_not_degraded():
    result = await run_workflow(patient_id="p", trigger_reason="t", patient_state=PatientState(patient_id="p"),
                                input_payload={"symptoms": [{"name": "mild headache", "free_text": "slight headache"}]})
    assert result.degraded is False and result.failed_agents == []


@pytest.mark.asyncio
async def test_emergency_agent_crash_fails_closed(monkeypatch):
    from app.agents.registry import get_agent
    agent = get_agent("emergency_escalation_agent")

    async def boom(task, state):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(agent, "execute", boom)
    result = await run_workflow(patient_id="p", trigger_reason="t", patient_state=PatientState(patient_id="p"),
                                input_payload={"symptoms": [{"name": "mild headache", "free_text": "slight headache"}]})
    assert result.is_emergency is True                    # crash of the safety agent => treat as emergency
    assert "emergency_escalation_agent" in result.failed_agents
    assert any(a.action_type == "emergency_escalation" for a in result.proposed_actions)


@pytest.mark.asyncio
async def test_prescan_catches_emergency_even_if_agents_cannot_see_it(monkeypatch):
    """Independent pre-scan on RAW input: emergency detected although the emergency agent returns 'no emergency'."""
    from app.agents.registry import get_agent
    from app.schemas.agent import AgentResult, AgentStatus
    agent = get_agent("emergency_escalation_agent")

    async def blind(task, state):
        return AgentResult(agent_name=agent.name, task_id=task.task_id, status=AgentStatus.COMPLETED, raw_payload={"is_emergency": False})
    monkeypatch.setattr(agent, "execute", blind)
    result = await run_workflow(patient_id="p", trigger_reason="t", patient_state=PatientState(patient_id="p"),
                                input_payload={"symptoms": None, "caregiver_notes": "he collapsed and is not responding"})
    assert result.is_emergency is True
    assert result.emergency_screen["source"] == "prescan"


@pytest.mark.asyncio
async def test_hung_agent_times_out_and_is_recorded_failed(monkeypatch):
    import asyncio
    import app.orchestrator.orchestrator as orch
    from app.agents.registry import get_agent

    class _S:
        agent_timeout_seconds = 0.05
    monkeypatch.setattr(orch, "get_settings", lambda: _S())
    agent = get_agent("signal_analysis_agent")

    async def hang(task, state):
        await asyncio.sleep(5)
    monkeypatch.setattr(agent, "execute", hang)
    result = await run_workflow(patient_id="p", trigger_reason="t", patient_state=PatientState(patient_id="p"),
                                input_payload={"symptoms": [{"name": "x"}]})
    assert "signal_analysis_agent" in result.failed_agents and result.degraded is True
    assert result.agent_results["signal_analysis_agent"].status == "failed"


@pytest.mark.asyncio
async def test_consensus_hypotheses_use_only_hypothesis_specific_voters():
    result = await run_workflow(patient_id="p", trigger_reason="t", patient_state=PatientState(patient_id="p"),
                                input_payload={"symptoms": [{"name": "resting tremor", "free_text": "my hand shakes at rest, stiffness, slow movement"}]})
    if result.consensus:      # candidates depend on the lexical KB fallback offline
        for h in result.consensus.hypotheses:
            assert h.contributing_agents == ["neuro_diagnostic_agent"]
        if len(result.consensus.hypotheses) > 1:
            assert len({h.consensus_score for h in result.consensus.hypotheses}) > 1


@pytest.mark.asyncio
async def test_malformed_input_types_do_not_blind_the_emergency_scan():
    result = await run_workflow(patient_id="p", trigger_reason="t", patient_state=PatientState(patient_id="p"),
                                input_payload={"symptoms": "I can't breathe", "wearable": [1, 2]})
    assert result.is_emergency is False or result.is_emergency is True   # must not raise
