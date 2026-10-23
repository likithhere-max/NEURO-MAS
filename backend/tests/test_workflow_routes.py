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
    def update(self, payload): self.op, self.payload = "update", payload; return self
    def delete(self): self.op = "delete"; return self
    def limit(self, *a, **k): return self
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
        if self.op == "update":
            if self.name in self.fail:
                raise RuntimeError(f"simulated DB failure on {self.name}")
            for r in rows:
                r.update(copy.deepcopy(self.payload))
        if self.op == "delete":
            self.db[self.name] = [r for r in self.db.get(self.name, []) if r not in rows]
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


def test_duplicate_submission_without_key_creates_two_workflows_by_design(setup):
    """Without an Idempotency-Key the server cannot know a POST is a retry; clients that retry must send a key."""
    db, _ = setup
    _post("I can't breathe"); _post("I can't breathe")
    assert len(db["workflows"]) == 2


# ---------------- Phase 2: idempotency, durable state, emergency checkpoint ----------------
KEY = "retry-key-0001"


def _post_key(text, key=KEY):
    return client.post("/api/v1/workflows/neurological-assessment/pat1", headers={"Idempotency-Key": key}, json={
        "trigger_reason": "t", "observations": {"symptoms": [{"name": "symptom", "free_text": text}]}})


def test_same_idempotency_key_returns_stored_result_and_creates_no_duplicates(setup):
    db, _ = setup
    first = _post_key("I can't breathe").json()
    n_actions, n_alerts, n_events = len(db["actions"]), len(db["alerts"]), len(db["agent_event_log"])
    again = _post_key("I can't breathe")
    assert again.status_code == 200
    body = again.json()
    assert body["idempotent_replay"] is True and body["workflow_id"] == first["workflow_id"]
    assert len(db["workflows"]) == 1
    assert (len(db["actions"]), len(db["alerts"]), len(db["agent_event_log"])) == (n_actions, n_alerts, n_events)
    assert body["is_emergency"] is True            # the stored emergency result is replayed, not lost


def test_different_keys_or_patients_do_not_collide(setup):
    db, _ = setup
    a = _post_key("mild headache", "key-aaaaaaaa").json()["workflow_id"]
    b = _post_key("mild headache", "key-bbbbbbbb").json()["workflow_id"]
    assert a != b and len(db["workflows"]) == 2


@pytest.mark.parametrize("bad", ["short", "has space in it", "x" * 129, "semi;colon-key"])
def test_malformed_idempotency_key_is_rejected(setup, bad):
    r = client.post("/api/v1/workflows/neurological-assessment/pat1", headers={"Idempotency-Key": bad},
                    json={"trigger_reason": "t", "observations": {"symptoms": []}})
    assert r.status_code == 422


def test_completed_run_is_marked_persist_state_complete(setup):
    db, _ = setup
    _post_key("mild headache")
    assert db["workflows"][0]["result_json"]["persist_state"] == "complete"


def test_child_table_failure_leaves_an_incomplete_marker_and_retry_repairs_it(setup):
    db, holder = setup
    holder["fail"] = ("actions",)
    r = _post_key("I can't breathe")                      # emergency: result is still returned
    assert r.status_code == 200 and r.json()["persisted"] is False and r.json()["is_emergency"] is True
    # the update on 'workflows' is not in the failure set, so the marker records the failure honestly
    assert db["workflows"][0]["result_json"]["persist_state"] == "incomplete"
    holder["fail"] = ()
    again = _post_key("I can't breathe")                  # not "complete" -> NOT a replay: rewrites children
    assert again.status_code == 200 and again.json().get("idempotent_replay") is None
    assert again.json()["persisted"] is True and len(db["workflows"]) == 1
    assert db["workflows"][0]["result_json"]["persist_state"] == "complete"
    ids = [a["id"] for a in db["actions"]]
    assert len(ids) == len(set(ids))


def test_emergency_alert_is_written_before_agents_run_even_if_everything_else_fails(setup, monkeypatch):
    """Checkpoint durability: the pre-scan alert survives total persistence failure of the final write."""
    db, holder = setup
    holder["fail"] = ("workflows", "agent_event_log", "audit_events", "actions")
    r = _post_key("my face is drooping and my speech is slurred")
    assert r.status_code == 200 and r.json()["persisted"] is False and r.json()["is_emergency"] is True
    pre = [a for a in db["alerts"] if str(a["reason"]).startswith("EMERGENCY PRE-SCAN")]
    assert len(pre) == 1 and pre[0]["severity"] == "critical"


def test_checkpoint_is_idempotent_across_retries(setup):
    db, holder = setup
    holder["fail"] = ("workflows",)
    _post_key("I can't breathe"); _post_key("I can't breathe")
    assert len([a for a in db["alerts"] if str(a["reason"]).startswith("EMERGENCY PRE-SCAN")]) == 1


def test_non_emergency_run_writes_no_checkpoint_alert(setup):
    db, _ = setup
    _post_key("mild headache")
    assert not [a for a in db.get("alerts", []) if str(a["reason"]).startswith("EMERGENCY PRE-SCAN")]


def test_response_exposes_typed_consensus_labels_and_run_health(setup):
    body = _post("my hand shakes at rest, stiffness and slow movement").json()
    assert "timed_out_agents" in body and "deadline_exceeded" in body
    if body["consensus"]:
        c = body["consensus"]
        assert c["calibrated"] is False and c["probability"] is None and c["score_kind"] == "uncalibrated_support_index"


# ---------------- Phase 2: notification dispatch wired after persistence (MOCK provider) ----------------
from app.services import notification_dispatch as nd, notification_service as ns  # noqa: E402
from tests.helpers_notifications import MockProvider  # noqa: E402


def _install_dispatcher(monkeypatch, provider, db):
    store = nd.InMemoryNotificationStore()
    events = []

    def build(admin, escalate=None, audit=None):
        async def sleep(d): pass
        return nd.NotificationDispatcher(provider, store, audit=events.append, escalate=escalate, max_attempts=2,
                                         attempt_timeout=1.0, backoff_base=0.0, sleep=sleep)
    monkeypatch.setattr(ns, "build_dispatcher", build)
    monkeypatch.setattr(wf_routes, "get_supabase_admin_client", lambda: object(), raising=False)
    import app.core.supabase_client as sc
    monkeypatch.setattr(sc, "get_supabase_admin_client", lambda: object())
    monkeypatch.setattr(ns, "get_settings", lambda: type("S", (), {"notification_channel": "mock"})())
    return store, events


def test_emergency_action_is_dispatched_once_and_status_reflects_the_provider(setup, monkeypatch):
    db, _ = setup
    db["patients"][0] = {**db["patients"][0], "clinician_id": "doc"}
    p = MockProvider([("sent", "")])
    store, _ev = _install_dispatcher(monkeypatch, p, db)
    body = _post_key("I can't breathe").json()
    esc = [a for a in body["proposed_actions"] if a["action_type"] == "emergency_escalation"][0]
    assert esc["status"] == "sent_unconfirmed"               # accepted by provider, delivery NOT confirmed
    assert len(p.calls) == 1
    assert [a["status"] for a in db["actions"] if a["action_type"] == "emergency_escalation"] == ["sent_unconfirmed"]
    again = _post_key("I can't breathe").json()                # idempotent replay: nothing more is sent
    assert again["idempotent_replay"] is True and len(p.calls) == 1


def test_failed_delivery_raises_a_critical_alert_and_does_not_claim_execution(setup, monkeypatch):
    db, _ = setup
    p = MockProvider([("transient_error", "")])
    _install_dispatcher(monkeypatch, p, db)
    body = _post_key("I can't breathe").json()
    esc = [a for a in body["proposed_actions"] if a["action_type"] == "emergency_escalation"][0]
    assert esc["status"] == "delivery_failed" and body["is_emergency"] is True
    assert any(str(a["reason"]).startswith("NOTIFICATION DELIVERY FAILED") and a["severity"] == "critical" for a in db["alerts"])


def test_no_channel_configured_sends_nothing_and_keeps_actions_proposed(setup):
    db, _ = setup
    body = _post_key("I can't breathe").json()
    esc = [a for a in body["proposed_actions"] if a["action_type"] == "emergency_escalation"][0]
    assert esc["status"] == "proposed"


def test_default_config_does_not_log_an_error_or_need_admin_credentials(setup, caplog):
    """NOTIFICATION_CHANNEL=none (default): the request path must not request admin credentials or log ERROR."""
    import logging
    with caplog.at_level(logging.ERROR, logger="neuromas.workflows"):
        assert _post("mild headache").status_code == 200
    assert not [r for r in caplog.records if "notification_stage_failed" in r.getMessage()]
