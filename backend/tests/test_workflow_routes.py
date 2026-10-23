"""Route -> orchestrator -> persistence integration, with Supabase replaced by an in-memory fake.
Verifies OUR code path (validation, orchestration, persistence ordering, failure handling, response contract).
It does NOT validate real Supabase RLS/schema."""
import copy

import pytest
from fastapi.testclient import TestClient

import app.api.routes.workflows as wf_routes
from app.main import app
from app.services.auth import get_current_user

client = TestClient(app)


class _Res:
    def __init__(self, data): self.data = data


class FakeQuery:
    def __init__(self, name, db, fail_tables):
        self.name, self.db, self.fail, self.op, self.payload, self.filters = name, db, fail_tables, "select", None, []
    def select(self, *a, **k): self.op = "select"; return self
    def insert(self, payload): self.op, self.payload = "insert", payload; return self
    def eq(self, c, v): self.filters.append((c, v)); return self
    def order(self, *a, **k): return self
    def execute(self):
        if self.op == "insert":
            if self.name in self.fail:
                raise RuntimeError(f"simulated DB failure on {self.name}")
            rows = self.payload if isinstance(self.payload, list) else [self.payload]
            stored = [{"id": r.get("id", f"{self.name}-{len(self.db.setdefault(self.name, []))+i}"), **copy.deepcopy(r)} for i, r in enumerate(rows)]
            self.db.setdefault(self.name, []).extend(stored)
            return _Res(stored)
        rows = [r for r in self.db.get(self.name, []) if all(r.get(c) == v for c, v in self.filters)]
        return _Res(rows)


class FakeClient:
    def __init__(self, db, fail=()): self.db, self.fail = db, set(fail)
    def table(self, name): return FakeQuery(name, self.db, self.fail)


PATIENT = {"id": "pat1", "full_name": "T", "age": 60, "sex": "male", "preferred_language": None, "user_id": "u", "clinician_id": "doc"}


@pytest.fixture
def setup(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: {"sub": "doc", "_access_token": "t"}
    db = {"patients": [PATIENT]}
    holder = {"fail": ()}
    monkeypatch.setattr(wf_routes, "get_authed_client", lambda token: FakeClient(db, holder["fail"]))
    yield db, holder
    app.dependency_overrides.clear()


def _post(text, name="symptom"):
    return client.post("/api/v1/workflows/neurological-assessment/pat1", json={
        "trigger_reason": "t", "observations": {"symptoms": [{"name": name, "free_text": text}]}})


def test_emergency_flows_end_to_end_and_is_persisted(setup):
    db, _ = setup
    r = _post("my face is drooping and my speech is slurred")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_emergency"] is True and body["persisted"] is True
    esc = [a for a in body["proposed_actions"] if a["action_type"] == "emergency_escalation"][0]
    assert esc["status"] == "proposed"            # no delivery channel -> never claims "executed"
    assert any(a["action_type"] == "emergency_escalation" for a in db["actions"])
    assert any(a["severity"] == "critical" for a in db["alerts"])       # an alert row exists for clinicians
    assert db["workflows"][0]["is_emergency"] is True
    assert "degraded" in db["workflows"][0]["result_json"]


def test_persistence_failure_does_not_swallow_an_emergency(setup):
    _, holder = setup
    holder["fail"] = ("workflows",)
    r = _post("I can't breathe")
    assert r.status_code == 200
    body = r.json()
    assert body["is_emergency"] is True and body["persisted"] is False


def test_persistence_failure_on_non_emergency_is_a_clean_500(setup):
    _, holder = setup
    holder["fail"] = ("workflows",)
    r = _post("mild headache")
    assert r.status_code == 500 and "simulated" not in r.text


def test_response_contract_for_frontend(setup):
    r = _post("mild headache")
    assert r.status_code == 200
    body = r.json()
    for key in ("workflow_id", "patient_id", "status", "is_emergency", "consensus", "agent_results", "proposed_actions",
                "audit_trail", "persisted", "degraded", "failed_agents"):
        assert key in body, key
    assert body["is_emergency"] is False and body["degraded"] is False
    assert {"data_intake_agent", "emergency_escalation_agent", "neuro_diagnostic_agent"} <= set(body["agent_results"])


def test_unknown_patient_is_404(setup):
    r = client.post("/api/v1/workflows/neurological-assessment/nope", json={"trigger_reason": "t", "observations": {"symptoms": []}})
    assert r.status_code == 404


@pytest.mark.parametrize("bad", [
    {"symptoms": [{"name": "x", "free_text": "a" * 5001}]},
    {"symptoms": "not a list"},
    {"symptoms": [{"free_text": "missing name"}]},
])
def test_invalid_payload_is_rejected_with_422(setup, bad):
    r = client.post("/api/v1/workflows/neurological-assessment/pat1", json={"trigger_reason": "t", "observations": bad})
    assert r.status_code == 422


def test_duplicate_submission_creates_two_workflows_documented_gap(setup):
    """No idempotency key exists: a retried POST creates a second workflow + alerts (see OUTSTANDING_ISSUES)."""
    db, _ = setup
    _post("I can't breathe"); _post("I can't breathe")
    assert len(db["workflows"]) == 2
