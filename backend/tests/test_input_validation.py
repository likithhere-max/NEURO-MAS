import pytest
from pydantic import ValidationError

from app.agents.data_intake import DataIntakeAgent
from app.agents.signal_analysis import SignalAnalysisAgent
from app.schemas.agent import AgentStatus, AgentTask
from app.schemas.api import (
    KnowledgeSearchIn, MedicationIn, MetricReadingIn, PatientCreate, SymptomIn, UserCreate,
)
from app.schemas.patient_state import PatientState


@pytest.mark.parametrize("factory", [
    lambda: MedicationIn(name="x", adherence_pct_30d=60),          # percent given where a fraction is required
    lambda: MedicationIn(name="x", adherence_pct_30d=-0.1),
    lambda: MedicationIn(name="x", adherence_pct_30d=float("nan")),
    lambda: MetricReadingIn(metric="m", value=float("nan")),
    lambda: MetricReadingIn(metric="m", value=float("inf")),
    lambda: PatientCreate(full_name="x", age=-5),
    lambda: PatientCreate(full_name="x", age=900),
    lambda: PatientCreate(full_name=""),
    lambda: SymptomIn(name="x", free_text="a" * 5001),
    lambda: KnowledgeSearchIn(query="x", top_k=10_000),
    lambda: UserCreate(email="a@b.co", password="short"),
])
def test_invalid_api_inputs_rejected(factory):
    with pytest.raises(ValidationError):
        factory()


def test_valid_inputs_still_accepted():
    assert MedicationIn(name="levodopa", adherence_pct_30d=0.85).adherence_pct_30d == 0.85
    assert MedicationIn(name="levodopa").adherence_pct_30d is None
    assert MetricReadingIn(metric="tremor_amplitude", value=1.5).value == 1.5
    assert PatientCreate(full_name="A", age=0).age == 0 and PatientCreate(full_name="A", age=130).age == 130


def _t(payload):
    return AgentTask(workflow_id="w", patient_id="p", agent_name="x", reason="t", input_payload=payload)


async def test_intake_survives_none_and_wrong_types():
    r = await DataIntakeAgent().execute(_t({"symptoms": None, "wearable": None, "vitals": "oops", "medication_logs": 7}), PatientState(patient_id="p"))
    assert r.status == AgentStatus.COMPLETED
    assert any(w.startswith("invalid_input:") for w in r.warnings)


async def test_intake_modality_cannot_be_overridden_by_payload():
    r = await DataIntakeAgent().execute(_t({"wearable": {"modality": "HACKED", "tremor_amplitude": 1.0}}), PatientState(patient_id="p"))
    assert {o["modality"] for o in r.raw_payload["observations"]} == {"wearable"}


async def test_intake_rejects_implausible_and_non_finite_values():
    r = await DataIntakeAgent().execute(_t({"vitals": {"heart_rate": 900, "spo2": float("nan"), "temperature_c": 37.0}}), PatientState(patient_id="p"))
    vit = [o for o in r.raw_payload["observations"] if o["modality"] == "vital"][0]
    assert vit == {"temperature_c": 37.0, "modality": "vital"}
    assert len(r.raw_payload["invalid_inputs"]) == 2
    assert r.confidence < 1.0 and r.raw_payload["data_quality"] < r.raw_payload["completeness"]


async def test_intake_dedupes_identical_symptoms():
    r = await DataIntakeAgent().execute(_t({"symptoms": [{"name": "tremor"}] * 3}), PatientState(patient_id="p"))
    assert r.raw_payload["duplicates_removed"] == 2
    assert len([o for o in r.raw_payload["observations"] if o["modality"] == "symptom"]) == 1


async def test_intake_scans_name_when_free_text_is_benign():
    r = await DataIntakeAgent().execute(_t({"symptoms": [{"name": "face drooping", "free_text": "feeling fine"}]}), PatientState(patient_id="p"))
    assert r.raw_payload["urgency"] == "critical"


async def test_signal_agent_zero_baseline_does_not_crash():
    st = PatientState(patient_id="p", wearable_metrics={"tremor_amplitude_baseline": 0, "gait_speed_baseline": 0})
    r = await SignalAnalysisAgent().execute(_t({"wearable": {"tremor_amplitude": 1.0, "gait_speed_m_s": 0.9}}), st)
    assert r.status == AgentStatus.COMPLETED
    assert "tremor_baseline_missing_or_nonpositive_no_relative_comparison" in r.warnings
    assert r.confidence == 0.0              # nothing could be compared to a baseline


async def test_signal_agent_non_numeric_value_does_not_crash():
    st = PatientState(patient_id="p", wearable_metrics={"tremor_amplitude_baseline": 2})
    r = await SignalAnalysisAgent().execute(_t({"wearable": {"tremor_amplitude": "high", "gait_speed_m_s": float("nan")}}), st)
    assert "invalid_value:tremor_amplitude" in r.warnings and "invalid_value:gait_speed_m_s" in r.warnings


async def test_signal_agent_hand_calculated_thresholds():
    st = PatientState(patient_id="p", wearable_metrics={"tremor_amplitude_baseline": 2.0, "gait_speed_baseline": 1.0})
    r = await SignalAnalysisAgent().execute(_t({"wearable": {"tremor_amplitude": 3.1, "gait_speed_m_s": 0.79}}), st)
    labels = {f.label for f in r.findings}
    assert labels == {"tremor_increase", "gait_slowing"}          # 3.1 > 2.0*1.5=3.0 ; 0.79 < 1.0*0.8
    r = await SignalAnalysisAgent().execute(_t({"wearable": {"tremor_amplitude": 3.0, "gait_speed_m_s": 0.80}}), st)
    assert {f.label for f in r.findings} == {"tremor_amplitude", "gait_speed"}   # boundaries are exclusive
    assert r.confidence == 1.0


async def test_signal_agent_does_not_claim_to_evaluate_unanalysed_metrics():
    r = await SignalAnalysisAgent().execute(_t({"wearable": {"steps": 5000, "heart_rate_avg": 80}}), PatientState(patient_id="p"))
    assert r.status == AgentStatus.SKIPPED
    assert "Evaluated 0" in r.reasoning_summary
    assert any(w.startswith("recorded_but_not_analysed") for w in r.warnings)
