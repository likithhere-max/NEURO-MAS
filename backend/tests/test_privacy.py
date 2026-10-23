"""Consent gate, external-processing policy and redaction (Phase 2, F3). Gemini itself is replaced by a recorder:
these tests verify what WE send/withhold, not Google's behaviour or any legal compliance."""
import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

import app.ai.gemini_service as gs
import app.api.routes.workflows as wf_routes
from app.core import privacy
from app.core.config import Settings
from app.main import app
from app.orchestrator.orchestrator import run_workflow
from app.schemas.patient_state import PatientState
from app.services.auth import get_current_user


class Out(BaseModel):
    explanation: str = "ok"


class Sent:
    def __init__(self): self.calls = []


@pytest.fixture
def sent(monkeypatch):
    rec = Sent()

    class R:
        parsed = Out()
        usage_metadata = None

    def fake(model, contents, config):
        rec.calls.append({"contents": contents, "system": config.get("system_instruction")})
        return R()
    monkeypatch.setattr(gs, "_call_gemini", fake)
    return rec


def policy(monkeypatch, **kw):
    s = Settings(_env_file=None, **kw)
    monkeypatch.setattr(privacy, "get_settings", lambda: s)
    monkeypatch.setattr(gs, "get_settings", lambda: s)
    return s


# ------------------------------------------------------------------ decisions
def test_default_policy_blocks_without_scope_or_consent(monkeypatch, sent):
    policy(monkeypatch)
    with pytest.raises(gs.ExternalProcessingBlockedError, match="no_processing_scope"):
        gs.generate_structured("hello", Out)
    with privacy.processing_scope(purpose="t", consent=False):
        with pytest.raises(gs.ExternalProcessingBlockedError, match="consent_not_recorded"):
            gs.generate_structured("hello", Out)
    assert sent.calls == []                                   # nothing was sent


def test_blocked_error_is_a_gemini_error_so_existing_fallbacks_fire():
    assert issubclass(gs.ExternalProcessingBlockedError, gs.GeminiServiceError)


def test_consent_allows_and_system_scope_allows(monkeypatch, sent):
    policy(monkeypatch)
    with privacy.processing_scope(purpose="t", consent=True):
        assert gs.generate_structured("symptom text", Out).explanation == "ok"
    with privacy.system_scope("kb_ingestion"):
        gs.generate_structured("kb text", Out)
    assert len(sent.calls) == 2


def test_disabled_mode_blocks_even_with_consent(monkeypatch, sent):
    policy(monkeypatch, external_processing_mode="disabled")
    with privacy.processing_scope(purpose="t", consent=True):
        with pytest.raises(gs.ExternalProcessingBlockedError, match="disabled_by_policy"):
            gs.generate_structured("x", Out)
    assert sent.calls == []


def test_invalid_mode_fails_closed(monkeypatch, sent):
    s = policy(monkeypatch)
    object.__setattr__(s, "external_processing_mode", "yolo")
    with privacy.processing_scope(purpose="t", consent=True):
        with pytest.raises(gs.ExternalProcessingBlockedError, match="invalid_external_processing_mode"):
            gs.generate_structured("x", Out)


def test_consent_requirement_can_be_switched_off_for_synthetic_deployments(monkeypatch, sent):
    policy(monkeypatch, require_patient_consent=False)
    gs.generate_structured("synthetic", Out)
    assert len(sent.calls) == 1


def test_embedding_calls_obey_the_same_gate(monkeypatch):
    policy(monkeypatch)
    monkeypatch.setattr(gs, "_call_embed", lambda *a: (_ for _ in ()).throw(AssertionError("must not be called")))
    with pytest.raises(gs.ExternalProcessingBlockedError):
        gs.embed_text("my symptoms")


def test_scope_propagates_into_worker_threads(monkeypatch, sent):
    policy(monkeypatch)

    async def go():
        with privacy.processing_scope(purpose="t", consent=True):
            return await asyncio.to_thread(gs.generate_structured, "hi", Out)
    assert asyncio.run(go()).explanation == "ok"


# ------------------------------------------------------------------ redaction
@pytest.mark.parametrize("raw,label", [
    ("mail me at jane.doe@example.com please", "email"),
    ("see https://clinic.example/patient/123", "url"),
    ("ssn 123-45-6789", "ssn_like"),
    ("born 1961-04-02 and seen 04/05/2026", "date"),
    ("call (415) 555-0134 or +1 415 555 0134", "phone"),
    ("MRN 12345678 on file", "long_id"),
])
def test_identifier_patterns_are_masked(raw, label):
    out, counts = privacy.redact_text(raw)
    assert label in counts and f"[{label.upper()}]" in out


@pytest.mark.parametrize("clinical", [
    "tremor 4.5 Hz, amplitude 5.1 5.2 5.3 5.4", "levodopa 25/250 mg three times daily", "gait speed 0.82 m/s",
    "blood pressure 142/91, heart rate 76", "onset 3 weeks ago, severity 7/10", "readings 5.1, 5.3, 5.2, 5.4, 5.0",
])
def test_clinical_numbers_are_not_mangled(clinical):
    out, counts = privacy.redact_text(clinical)
    assert out == clinical and counts == {}


def test_known_patient_name_parts_are_masked_case_insensitively():
    out, counts = privacy.redact_text("Jane Doe says jane has a tremor; DOE's hand shakes", ["Jane Doe"])
    assert "jane" not in out.lower() and "doe" not in out.lower() and counts["patient_name"] >= 3


def test_redacted_text_is_what_leaves_and_audit_events_hold_no_text(monkeypatch, sent):
    policy(monkeypatch)
    with privacy.processing_scope(purpose="t", consent=True, identifiers=["Jane Doe"]) as ctx:
        gs.generate_structured("Jane Doe (jane@x.org) has a tremor since 2026-10-01", Out, system_instruction="Hi Jane")
    call = sent.calls[0]
    assert "Jane" not in call["contents"] and "jane@x.org" not in call["contents"] and "2026-10-01" not in call["contents"]
    assert "Jane" not in call["system"]
    ev = ctx.events[0]
    assert ev["allowed"] and ev["redactions"]["patient_name"] >= 1 and ev["redactions"]["email"] == 1
    assert "Jane" not in str(ev)


def test_unredacted_mode_sends_text_as_is(monkeypatch, sent):
    policy(monkeypatch, external_processing_mode="unredacted")
    with privacy.processing_scope(purpose="t", consent=True, identifiers=["Jane"]):
        gs.generate_structured("Jane has a tremor", Out)
    assert sent.calls[0]["contents"] == "Jane has a tremor"


# ------------------------------------------------------------------ workflow integration
SYMPTOMS = {"symptoms": [{"name": "resting tremor", "free_text": "Jane Doe: my hand shakes at rest, call 415-555-0134"}]}


@pytest.mark.asyncio
async def test_workflow_without_consent_sends_nothing_and_reports_it(monkeypatch, sent):
    policy(monkeypatch)
    r = await run_workflow(patient_id="p", trigger_reason="t", input_payload=SYMPTOMS,
                           patient_state=PatientState(patient_id="p", redaction_identifiers=["Jane Doe"]))
    assert sent.calls == []
    summ = [e for e in r.audit_events if e["event"] == "external_processing_summary"][0]["detail"]
    assert summ["calls_allowed"] == 0 and summ["calls_blocked"] >= 1
    assert "patient_consent_not_recorded" in summ["blocked_reasons"]
    assert r.status == "completed"                                       # templates carried the run


@pytest.mark.asyncio
async def test_workflow_with_consent_redacts_what_it_sends(monkeypatch, sent):
    policy(monkeypatch)
    monkeypatch.setattr("app.rag.knowledge_retriever._semantic_search_sync", lambda q, k: (_ for _ in ()).throw(gs.GeminiServiceError("off")))
    r = await run_workflow(patient_id="p", trigger_reason="t", input_payload=SYMPTOMS,
                           patient_state=PatientState(patient_id="p", external_ai_consent=True, redaction_identifiers=["Jane Doe"]))
    blob = " ".join(c["contents"] for c in sent.calls)
    assert sent.calls and "Jane" not in blob and "415-555-0134" not in blob
    summ = [e for e in r.audit_events if e["event"] == "external_processing_summary"][0]["detail"]
    assert summ["calls_allowed"] == len(sent.calls)


# ------------------------------------------------------------------ routes
client = TestClient(app)
PATIENT = {"id": "pat1", "full_name": "Jane Doe", "age": 60, "sex": "female", "user_id": "u", "clinician_id": "doc"}


class _Res:
    def __init__(self, d): self.data = d


class _Q:
    def __init__(self, rows): self.rows = rows
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def insert(self, p): self.rows = p if isinstance(p, list) else [p]; return self
    def update(self, p): return self
    def delete(self): return self
    def execute(self): return _Res(self.rows)


class _C:
    def __init__(self, patient): self.patient = patient
    def table(self, name): return _Q([self.patient] if name == "patients" else [])


def _symptom_post(monkeypatch, consent):
    app.dependency_overrides[get_current_user] = lambda: {"sub": "doc", "_access_token": "t"}
    monkeypatch.setattr(wf_routes, "get_authed_client", lambda t: _C({**PATIENT, "external_ai_consent": consent}))
    policy(monkeypatch)
    called = []

    async def fake_extract(payload):
        called.append(1)
        raise wf_routes.SymptomExtractionError("stop here")
    monkeypatch.setattr(wf_routes, "extract_symptom_profile", fake_extract)
    try:
        r = client.post("/api/v1/workflows/symptom-assessment/pat1", json={"free_text": "my hand shakes"})
    finally:
        app.dependency_overrides.clear()
    return r, called


def test_symptom_assessment_without_consent_is_403_and_extraction_never_runs(monkeypatch):
    r, called = _symptom_post(monkeypatch, consent=False)
    assert r.status_code == 403 and "consent_not_recorded" in r.text and called == []


def test_symptom_assessment_with_consent_reaches_extraction(monkeypatch):
    r, called = _symptom_post(monkeypatch, consent=True)
    assert called == [1] and r.status_code == 502        # our stub raised SymptomExtractionError after the gate passed


# ------------------------------------------------------------------ consent endpoint
class _UpdQ(_Q):
    log = []
    def update(self, p): _UpdQ.log.append(p); return self


def _consent_put(monkeypatch, rows):
    import app.api.routes.patients as pr
    app.dependency_overrides[get_current_user] = lambda: {"sub": "doc", "_access_token": "t"}

    class C:
        def table(self, name): return _UpdQ(rows)
    monkeypatch.setattr(pr, "get_authed_client", lambda t: C())
    _UpdQ.log.clear()
    try:
        return client.put("/api/v1/patients/pat1/ai-consent", json={"granted": True})
    finally:
        app.dependency_overrides.clear()


def test_consent_endpoint_records_who_and_when(monkeypatch):
    r = _consent_put(monkeypatch, [PATIENT])
    assert r.status_code == 200 and r.json()["external_ai_consent"] is True
    row = _UpdQ.log[0]
    assert row["external_ai_consent"] is True and row["external_ai_consent_by"] == "doc" and row["external_ai_consent_at"]


def test_consent_endpoint_404_when_row_not_visible(monkeypatch):
    assert _consent_put(monkeypatch, []).status_code == 404


def test_consent_payload_must_be_boolean(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: {"sub": "doc", "_access_token": "t"}
    try:
        assert client.put("/api/v1/patients/pat1/ai-consent", json={"granted": "maybe"}).status_code == 422
    finally:
        app.dependency_overrides.clear()
