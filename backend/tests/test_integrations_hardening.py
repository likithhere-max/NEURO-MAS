"""Gemini retry policy, diagnostic query construction, knowledge-agent honesty."""
import pytest

import app.ai.gemini_service as gs
from app.agents.diagnostic import NeuroDiagnosticAgent
from app.agents.knowledge import KnowledgeAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState


class _APIErr(Exception):
    def __init__(self, code): super().__init__(f"api error {code}"); self.code = code


@pytest.mark.parametrize("exc,expected", [
    (_APIErr(503), True), (_APIErr(429), True), (_APIErr(500), True), (_APIErr(504), True),
    (_APIErr(400), False), (_APIErr(401), False), (_APIErr(403), False), (_APIErr(404), False),
    (ConnectionError("reset"), True), (TimeoutError(), True),
    (RuntimeError("GEMINI_API_KEY is not set"), False), (ValueError("bad"), False),
])
def test_only_transient_errors_are_retried(exc, expected):
    assert gs.is_transient_error(exc) is expected


def _real_call(monkeypatch, exc):
    """Exercise the REAL tenacity-decorated function (conftest stubs the module attribute; undo for these)."""
    import importlib.util
    # Load a PRIVATE copy of the module so the real, tenacity-decorated _call_gemini is exercised without
    # touching the shared module (conftest stubs that one; reloading it would leak un-stubbed code into other tests).
    spec = importlib.util.spec_from_file_location("gemini_service_private_copy", gs.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = {"n": 0}

    class _Models:
        def generate_content(self, **kw):
            calls["n"] += 1
            raise exc
    class _Client:
        models = _Models()
    monkeypatch.setattr(mod, "_get_client", lambda: _Client())
    mod._call_gemini.retry.sleep = lambda s: None       # no real backoff sleeping in tests
    return mod, calls


def test_permanent_error_is_not_retried(monkeypatch):
    mod, calls = _real_call(monkeypatch, _APIErr(400))
    with pytest.raises(_APIErr):
        mod._call_gemini("m", "c", {})
    assert calls["n"] == 1


def test_transient_error_is_retried_three_times(monkeypatch):
    mod, calls = _real_call(monkeypatch, _APIErr(503))
    with pytest.raises(_APIErr):
        mod._call_gemini("m", "c", {})
    assert calls["n"] == 3


def test_client_has_a_request_timeout():
    assert 0 < gs.REQUEST_TIMEOUT_MS <= 60_000


def _t(payload):
    return AgentTask(workflow_id="w", patient_id="p", agent_name="x", reason="t", input_payload=payload)


async def test_diagnostic_queries_with_free_text_not_the_placeholder_label(monkeypatch):
    import app.agents.diagnostic as dg
    seen = {}

    async def fake_match(terms, known, top_k=5):
        seen["terms"] = terms
        return []
    monkeypatch.setattr(dg.knowledge_retriever, "match_symptoms", fake_match)
    obs = [{"modality": "symptom", "name": "patient-reported symptoms (verbatim)", "free_text": "my hands shake when resting"},
           {"modality": "symptom", "name": "resting tremor", "free_text": ""}]
    await NeuroDiagnosticAgent().execute(_t({"observations": obs}), PatientState(patient_id="p"))
    assert "my hands shake when resting" in seen["terms"] and "resting tremor" in seen["terms"]
    assert "patient-reported symptoms (verbatim)" not in seen["terms"]


async def test_knowledge_agent_strength_and_confidence_are_derived_not_constant():
    cands = [{"disease_id": "parkinsons_disease", "preferred_name": "Parkinson's disease", "retrieval_score": 0.9},
             {"disease_id": "does_not_exist", "preferred_name": "Fake", "retrieval_score": 0.5}]
    from app.services.knowledge_base import get_knowledge_base
    real_id = get_knowledge_base().list_all()[0]["disease_id"]
    cands[0]["disease_id"] = real_id
    r = await KnowledgeAgent().execute(_t({"upstream_results": {"neuro_diagnostic_agent": {"raw_payload": {"candidate_conditions": cands}}}}), PatientState(patient_id="p"))
    assert r.confidence == 0.5                              # 1 of 2 candidates has a KB entry
    assert {e.strength for e in r.evidence} == {0.9}        # the candidate's own retrieval score, not 0.7
    assert "confidence_is_kb_coverage_not_probability" in r.warnings
    assert r.raw_payload["citations"] and all(c["source_url"] for c in r.raw_payload["citations"])
