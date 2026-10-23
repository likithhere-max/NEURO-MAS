"""Security regression tests. Supabase is replaced by in-memory fakes -- these verify OUR authorization logic,
not Supabase RLS (which cannot be tested without a live project; see docs/audit/SECURITY_PRIVACY_REPORT.md)."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.services.auth import get_current_user

client = TestClient(app)


class FakeQuery:
    def __init__(self, table, store):
        self.table_name, self.store, self.filters, self.payload, self.op = table, store, [], None, "select"

    def select(self, *a, **k): self.op = "select"; return self
    def update(self, payload): self.op, self.payload = "update", payload; return self
    def insert(self, payload): self.op, self.payload = "insert", payload; return self
    def eq(self, col, val): self.filters.append((col, val)); return self
    def order(self, *a, **k): return self

    def execute(self):
        rows = [r for r in self.store.get(self.table_name, []) if all(r.get(c) == v for c, v in self.filters)]
        res = MagicMock()
        if self.op == "update":
            for r in rows: r.update(self.payload)
        if self.op == "insert":
            row = {"id": "new-id", **self.payload}; self.store.setdefault(self.table_name, []).append(row); rows = [row]
        res.data = rows
        return res


class FakeClient:
    def __init__(self, store): self.store = store
    def table(self, name): return FakeQuery(name, self.store)


def _as_user(sub, store, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: {"sub": sub, "_access_token": "t"}
    import app.api.routes.alerts as alerts
    monkeypatch.setattr(alerts, "get_authed_client", lambda token: FakeClient(store))


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _store():
    return {"profiles": [{"id": "doc", "role": "clinician"}, {"id": "pat", "role": "patient"}],
            "alerts": [{"id": "a1", "status": "open", "patient_id": "pat"}],
            "actions": [{"id": "x1", "status": "pending_approval"}, {"id": "x2", "status": "proposed"}, {"id": "x3", "status": "executed"}]}


def test_patient_cannot_acknowledge_alert(monkeypatch):
    store = _store(); _as_user("pat", store, monkeypatch)
    assert client.post("/api/v1/alerts/a1/acknowledge", json={}).status_code == 403
    assert store["alerts"][0]["status"] == "open"


def test_clinician_can_acknowledge_alert(monkeypatch):
    store = _store(); _as_user("doc", store, monkeypatch)
    assert client.post("/api/v1/alerts/a1/acknowledge", json={}).status_code == 200
    assert store["alerts"][0]["status"] == "acknowledged"


def test_patient_cannot_approve_action(monkeypatch):
    store = _store(); _as_user("pat", store, monkeypatch)
    assert client.post("/api/v1/alerts/actions/x1/approve").status_code == 403
    assert store["actions"][0]["status"] == "pending_approval"


def test_clinician_can_only_approve_pending_actions(monkeypatch):
    store = _store(); _as_user("doc", store, monkeypatch)
    assert client.post("/api/v1/alerts/actions/x1/approve").status_code == 200
    assert client.post("/api/v1/alerts/actions/x2/approve").status_code == 404     # 'proposed' is not awaiting approval
    assert client.post("/api/v1/alerts/actions/x3/approve").status_code == 404     # already executed: no re-approval
    assert store["actions"][1]["status"] == "proposed"


# ---- signup ----

def _patch_supabase(monkeypatch):
    import app.api.routes.auth as auth
    admin, public = MagicMock(), MagicMock()
    admin.auth.admin.create_user.return_value = MagicMock(user=MagicMock(id="uid"))
    public.auth.sign_in_with_password.return_value = MagicMock(session=MagicMock(access_token="tok"), user=MagicMock(id="uid"))
    monkeypatch.setattr(auth, "get_supabase_admin_client", lambda: admin)
    monkeypatch.setattr(auth, "new_public_client", lambda: public)
    monkeypatch.setattr(auth, "get_authed_client", lambda token: MagicMock())
    return admin


def _settings(monkeypatch, **kw):
    import app.api.routes.auth as auth
    monkeypatch.setattr(auth, "get_settings", lambda: Settings(_env_file=None, **kw))


BODY = {"email": "a@example.com", "password": "longenough1", "full_name": "A", "role": "clinician"}


def test_clinician_signup_blocked_in_production_without_invite(monkeypatch):
    _patch_supabase(monkeypatch); _settings(monkeypatch, environment="production")
    r = client.post("/api/v1/auth/signup", json=BODY)
    assert r.status_code == 403


def test_clinician_signup_requires_matching_invite_code(monkeypatch):
    admin = _patch_supabase(monkeypatch); _settings(monkeypatch, environment="production", clinician_invite_code="s3cret")
    assert client.post("/api/v1/auth/signup", json={**BODY, "invite_code": "wrong"}).status_code == 403
    assert client.post("/api/v1/auth/signup", json=BODY).status_code == 403
    admin.auth.admin.create_user.assert_not_called()
    assert client.post("/api/v1/auth/signup", json={**BODY, "invite_code": "s3cret"}).status_code == 200


def test_clinician_signup_allowed_in_development_for_demos(monkeypatch):
    _patch_supabase(monkeypatch); _settings(monkeypatch, environment="development")
    assert client.post("/api/v1/auth/signup", json=BODY).status_code == 200


def test_patient_signup_never_needs_invite(monkeypatch):
    _patch_supabase(monkeypatch); _settings(monkeypatch, environment="production")
    assert client.post("/api/v1/auth/signup", json={**BODY, "role": "patient"}).status_code == 200


def test_signup_does_not_leak_backend_exception_text(monkeypatch):
    admin = _patch_supabase(monkeypatch); _settings(monkeypatch, environment="development")
    admin.auth.admin.create_user.side_effect = Exception("secret-internal-detail: postgres://u:p@host")
    r = client.post("/api/v1/auth/signup", json={**BODY, "role": "patient"})
    assert r.status_code == 400 and "secret-internal-detail" not in r.text


def test_failed_profile_insert_removes_orphan_auth_user(monkeypatch):
    import app.api.routes.auth as auth
    admin = _patch_supabase(monkeypatch); _settings(monkeypatch, environment="development")
    bad = MagicMock(); bad.table.return_value.insert.return_value.execute.side_effect = RuntimeError("rls")
    monkeypatch.setattr(auth, "get_authed_client", lambda token: bad)
    r = client.post("/api/v1/auth/signup", json={**BODY, "role": "patient"})
    assert r.status_code == 500
    admin.auth.admin.delete_user.assert_called_once_with("uid")


def test_login_without_profile_is_a_clean_403_not_a_500(monkeypatch):
    import app.api.routes.auth as auth
    _patch_supabase(monkeypatch)
    empty = MagicMock(); empty.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    monkeypatch.setattr(auth, "get_authed_client", lambda token: empty)
    assert client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "x"}).status_code == 403


def test_auth_endpoints_are_rate_limited(monkeypatch):
    import app.api.routes.auth as auth
    from app.core import ratelimit
    _patch_supabase(monkeypatch)
    monkeypatch.setattr(ratelimit, "get_settings", lambda: Settings(_env_file=None, auth_rate_limit_per_minute=3))
    bad = MagicMock(); bad.auth.sign_in_with_password.side_effect = Exception("bad")
    monkeypatch.setattr(auth, "new_public_client", lambda: bad)
    codes = [client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "x"}).status_code for _ in range(5)]
    assert codes == [401, 401, 401, 429, 429]


# ---- CORS ----

def test_credentials_are_not_combined_with_wildcard_origin():
    r = client.options("/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-credentials") != "true"


def test_protected_routes_reject_missing_token():
    assert client.get("/api/v1/patients").status_code in (401, 403)
    assert client.post("/api/v1/alerts/a1/acknowledge", json={}).status_code in (401, 403)
