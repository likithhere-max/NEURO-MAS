import pytest

from app.agents.longitudinal import LongitudinalAgent
from app.agents.personalization import PersonalizationAgent
from app.core.metric_semantics import classify_change, metric_polarity
from app.schemas.agent import AgentTask
from app.schemas.patient_state import LongitudinalTrend, PatientState


@pytest.mark.parametrize("metric,pol", [
    ("tremor_amplitude", "higher_worse"), ("seizure_frequency_per_week", "higher_worse"), ("UPDRS_part3", "higher_worse"),
    ("gait_speed_m_s", "lower_worse"), ("moca_score", "lower_worse"), ("medication_adherence_pct", "lower_worse"),
    ("balance_score", "lower_worse"), ("sleep_hours", "lower_worse"), ("something_new", "unknown"),
])
def test_polarity_registry(metric, pol):
    assert metric_polarity(metric) == pol


@pytest.mark.parametrize("metric,base,cur,expected", [
    ("gait_speed_m_s", 1.2, 0.7, "deteriorating"),      # -41.7%: was 'improving' before the fix
    ("moca_score", 28, 18, "deteriorating"),
    ("gait_speed_m_s", 1.0, 1.4, "improving"),
    ("tremor_amplitude", 2.0, 3.5, "deteriorating"),    # +75%
    ("tremor_amplitude", 2.0, 1.2, "improving"),
    ("tremor_amplitude", 2.0, 2.05, "stable"),          # +2.5% inside the 5% band
    ("tremor_amplitude", 2.0, 2.3, "stable"),           # +15%: below the 20% flag threshold
    ("mystery_metric", 10, 14, "changed_direction_unknown"),   # never guessed
    ("mystery_metric", 10, 10.2, "stable"),
])
def test_classification_is_polarity_aware(metric, base, cur, expected):
    assert classify_change(metric, base, cur)["classification"] == expected


def test_exact_threshold_boundary_is_inclusive():
    # +20.0% exactly (1.0 -> 1.2) on a higher-is-worse metric counts as adverse.
    assert classify_change("tremor_amplitude", 1.0, 1.2)["classification"] == "deteriorating"


def test_zero_baseline_does_not_invent_a_percentage():
    r = classify_change("seizure_count_week", 0, 0.3)
    assert r["classification"] == "baseline_zero_undefined" and r["delta_pct"] is None and r["delta_abs"] == pytest.approx(0.3)
    assert classify_change("seizure_count_week", 0, 0)["classification"] == "stable"


def test_negative_baseline_uses_absolute_value_in_denominator():
    r = classify_change("mystery_metric", -10, -5)           # +50% of |baseline|
    assert r["delta_pct"] == pytest.approx(0.5)


def test_non_finite_values_are_invalid():
    assert classify_change("tremor_amplitude", float("nan"), 1.0)["classification"] == "invalid"
    assert classify_change("tremor_amplitude", None, 1.0)["classification"] == "unknown"


def _task(payload=None):
    return AgentTask(workflow_id="w", patient_id="p", agent_name="x", reason="t", input_payload=payload or {})


async def test_longitudinal_agent_end_to_end_polarity():
    state = PatientState(patient_id="p", longitudinal_trends=[
        LongitudinalTrend(metric="gait_speed_m_s", baseline_value=1.2, current_value=0.7),
        LongitudinalTrend(metric="moca_score", baseline_value=28, current_value=18),
        LongitudinalTrend(metric="tremor_amplitude", baseline_value=2.0, current_value=1.0),
        LongitudinalTrend(metric="mystery", baseline_value=1.0, current_value=2.0),
    ])
    r = await LongitudinalAgent().execute(_task(), state)
    assert set(r.raw_payload["deteriorating_metrics"]) == {"gait_speed_m_s", "moca_score"}
    assert r.raw_payload["improving_metrics"] == ["tremor_amplitude"]
    assert r.raw_payload["unclassified_metrics"] == ["mystery"]
    assert r.requires_human_review is True


async def test_personalization_never_says_advance_rehab_for_a_declining_patient():
    state = PatientState(patient_id="p", longitudinal_trends=[LongitudinalTrend(metric="gait_speed_m_s", baseline_value=1.2, current_value=0.7)])
    lr = await LongitudinalAgent().execute(_task(), state)
    pr = await PersonalizationAgent().execute(
        _task({"upstream_results": {"longitudinal_agent": {"raw_payload": lr.raw_payload}}}), state)
    plan = " ".join(pr.raw_payload["monitoring_plan"]).lower()
    assert "advance rehabilitation" not in plan and "increase monitoring frequency" in plan


async def test_anomaly_in_favourable_direction_is_not_high_severity():
    t = LongitudinalTrend(metric="tremor_amplitude", is_personal_anomaly=True, anomaly_z_score=-6.0, readings_available=8, anomaly_method="z_score")
    r = await LongitudinalAgent().execute(_task(), PatientState(patient_id="p", longitudinal_trends=[t]))
    f = [x for x in r.findings if x.label.startswith("personal_anomaly")][0]
    assert f.severity == "medium"
