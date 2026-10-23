"""Integration of app/ml/longitudinal.py through repository.build_patient_state -> LongitudinalAgent.

The Supabase client is a small in-memory fake (a MOCK: this verifies our wiring, not Supabase)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.agents.longitudinal import LongitudinalAgent
from app.schemas.agent import AgentStatus, AgentTask
from app.services.repository import build_patient_state


class _Q:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_):
        return self

    def eq(self, *_):
        return self

    def order(self, *_):
        return self

    def execute(self):
        class R:
            pass
        r = R()
        r.data = self.rows
        return r


class FakeClient:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return _Q(self.tables.get(name, []))


def _readings(metric, values, start=datetime(2026, 1, 1, tzinfo=timezone.utc)):
    return [{"metric": metric, "value": v, "recorded_at": (start + timedelta(days=i)).isoformat()} for i, v in enumerate(values)]


def _client(readings, metrics=None):
    return FakeClient({"metric_readings": readings, "longitudinal_metrics": metrics or []})


def _task():
    return AgentTask(workflow_id="wf", patient_id="p1", agent_name="longitudinal_agent", reason="t", input_payload={})


def test_state_carries_structured_report_and_compat_fields():
    rng = np.random.default_rng(1)
    vals = list(rng.normal(5, 0.5, 20))
    st = build_patient_state(_client(_readings("tremor_amplitude", vals)), {"id": "p1"})
    t = st.longitudinal_trends[0]
    assert t.anomaly_report["method_version"].startswith("longitudinal-")
    assert t.anomaly_report["is_diagnosis"] is False
    assert t.readings_available == 20
    assert t.anomaly_method == t.anomaly_report["method_version"]
    assert t.is_personal_anomaly == bool(t.anomaly_report["signals"])


def test_short_history_reports_insufficient_not_normal():
    st = build_patient_state(_client(_readings("tremor_amplitude", [5.0, 5.1])), {"id": "p1"})
    t = st.longitudinal_trends[0]
    assert t.anomaly_method == "insufficient_history"
    assert t.is_personal_anomaly is False
    assert t.anomaly_report["detection_status"] == "insufficient_data"


def test_missing_timestamps_are_handled_without_crash():
    rows = [{"metric": "tremor_amplitude", "value": float(v), "recorded_at": None} for v in range(10)]
    st = build_patient_state(_client(rows), {"id": "p1"})
    t = st.longitudinal_trends[0]
    assert t.anomaly_report["detection_status"] in ("invalid_input", "insufficient_data")
    assert t.is_personal_anomaly is False


@pytest.mark.asyncio
async def test_agent_flags_sustained_adverse_shift_and_lists_it(monkeypatch):
    monkeypatch.setattr("app.agents.longitudinal._explain_trends_with_gemini", _no_gemini)
    rng = np.random.default_rng(3)
    vals = list(rng.normal(5, 0.4, 14)) + list(rng.normal(8, 0.4, 14))
    st = build_patient_state(_client(_readings("tremor_amplitude", vals)), {"id": "p1"})
    res = await LongitudinalAgent().execute(_task(), st)
    assert res.status == AgentStatus.COMPLETED
    assert "tremor_amplitude" in res.raw_payload["personal_anomalies"]
    f = [x for x in res.findings if x.label == "personal_anomaly:tremor_amplitude"][0]
    assert f.severity == "high" and "not a diagnosis" in f.detail
    assert res.requires_human_review is True
    assert "tremor_amplitude" in res.raw_payload["anomaly_reports"]


@pytest.mark.asyncio
async def test_agent_reports_insufficient_history_as_warning(monkeypatch):
    monkeypatch.setattr("app.agents.longitudinal._explain_trends_with_gemini", _no_gemini)
    st = build_patient_state(_client(_readings("tremor_amplitude", [5.0, 5.2])), {"id": "p1"})
    res = await LongitudinalAgent().execute(_task(), st)
    assert any(w.startswith("insufficient_history_for_personal_analysis") for w in res.warnings)
    assert res.raw_payload["personal_anomalies"] == []


@pytest.mark.asyncio
async def test_agent_favourable_shift_is_medium_severity(monkeypatch):
    monkeypatch.setattr("app.agents.longitudinal._explain_trends_with_gemini", _no_gemini)
    rng = np.random.default_rng(5)
    vals = list(rng.normal(8, 0.4, 14)) + list(rng.normal(5, 0.4, 14))
    st = build_patient_state(_client(_readings("tremor_amplitude", vals)), {"id": "p1"})
    res = await LongitudinalAgent().execute(_task(), st)
    fs = [x for x in res.findings if x.label.startswith("personal_anomaly")]
    assert fs and all(x.severity == "medium" for x in fs)


async def _no_gemini(_):
    return None


@pytest.mark.asyncio
async def test_agent_surfaces_short_series_data_quality_warning(monkeypatch):
    monkeypatch.setattr("app.agents.longitudinal._explain_trends_with_gemini", _no_gemini)
    st = build_patient_state(_client(_readings("tremor_amplitude", [5.0, 5.2, 5.1, 5.0])), {"id": "p1"})
    res = await LongitudinalAgent().execute(_task(), st)
    assert any("short_series" in w for w in res.warnings)
