"""Agents report 'could not assess' with a typed status instead of COMPLETED-with-zero-confidence (Phase 2, D)."""
import pytest

from app.agents.diagnostic import NeuroDiagnosticAgent
from app.agents.longitudinal import LongitudinalAgent
from app.agents.signal_analysis import SignalAnalysisAgent
from app.core.consensus import compute_evidence_consensus
from app.schemas.agent import AgentStatus, AgentTask
from app.schemas.patient_state import PatientState


def _task(payload, agent="x"):
    return AgentTask(workflow_id="wf1", patient_id="p1", agent_name=agent, reason="test", input_payload=payload)


@pytest.mark.asyncio
async def test_longitudinal_with_no_history_is_insufficient_data_not_completed():
    r = await LongitudinalAgent().execute(_task({}), PatientState(patient_id="p1"))
    assert r.status == AgentStatus.INSUFFICIENT_DATA
    assert "no_baseline_history_available" in r.warnings
    assert r.hypothesis_evidence == []


@pytest.mark.asyncio
async def test_diagnostic_with_no_symptoms_is_insufficient_data_and_casts_no_vote():
    r = await NeuroDiagnosticAgent().execute(_task({"observations": []}), PatientState(patient_id="p1"))
    assert r.status == AgentStatus.INSUFFICIENT_DATA
    o = compute_evidence_consensus([r], {"neuro_diagnostic_agent": 0.65}, 0.9)
    assert o.top_hypothesis is None and o.non_voting_agents == {"neuro_diagnostic_agent": "insufficient_data"}


@pytest.mark.asyncio
async def test_signal_agent_with_only_unsupported_modalities_abstains():
    r = await SignalAnalysisAgent().execute(_task({"wearable": {"eeg_raw": [1, 2, 3]}}), PatientState(patient_id="p1"))
    assert r.status == AgentStatus.ABSTAINED
    assert any("MODEL_NOT_AVAILABLE" in w for w in r.warnings)
    assert r.findings == []
