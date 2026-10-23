import pytest

from app.agents.data_intake import DataIntakeAgent
from app.agents.medication import MedicationSafetyAgent
from app.agents.signal_analysis import SignalAnalysisAgent
from app.schemas.agent import AgentStatus, AgentTask
from app.schemas.patient_state import MedicationState, PatientState


def _task(payload):
    return AgentTask(workflow_id="wf1", patient_id="p1", agent_name="x", reason="test", input_payload=payload)


@pytest.mark.asyncio
async def test_data_intake_flags_emergency_from_free_text():
    agent = DataIntakeAgent()
    state = PatientState(patient_id="p1")
    result = await agent.execute(_task({
        "symptoms": [{"name": "seizure", "free_text": "status epilepticus, not stopping"}],
    }), state)
    assert result.raw_payload["urgency"] == "critical"
    assert result.requires_human_review is True


@pytest.mark.asyncio
async def test_data_intake_reports_missing_data():
    agent = DataIntakeAgent()
    state = PatientState(patient_id="p1")
    result = await agent.execute(_task({}), state)
    assert "missing_data:symptoms" in result.warnings
    assert result.raw_payload["data_quality"] == 0.0


@pytest.mark.asyncio
async def test_medication_agent_never_proposes_dose_change():
    agent = MedicationSafetyAgent()
    state = PatientState(
        patient_id="p1",
        medication_state=MedicationState(active_medications=[{"name": "levodopa"}, {"name": "haloperidol"}]),
    )
    result = await agent.execute(_task({}), state)
    assert result.raw_payload["action_restriction"] == "MEDICATION_CHANGES_REQUIRE_CLINICIAN_AUTHORIZATION"
    assert result.requires_human_review is True
    assert len(result.raw_payload["interactions"]) == 1
    # Confirm no key in the payload implies a dose/prescription action.
    assert not any("dose" in str(f.detail).lower() and "change" in str(f.detail).lower() for f in result.findings)


@pytest.mark.asyncio
async def test_signal_agent_returns_model_not_available_for_unsupported_modality():
    agent = SignalAnalysisAgent()
    state = PatientState(patient_id="p1")
    result = await agent.execute(_task({"wearable": {"eeg_raw": [1, 2, 3]}}), state)
    assert result.status == AgentStatus.ABSTAINED  # Phase 2: nothing was assessed, so it is not COMPLETED
    assert any("MODEL_NOT_AVAILABLE" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_signal_agent_skips_cleanly_with_no_data():
    agent = SignalAnalysisAgent()
    state = PatientState(patient_id="p1")
    result = await agent.execute(_task({}), state)
    assert result.status == AgentStatus.SKIPPED
