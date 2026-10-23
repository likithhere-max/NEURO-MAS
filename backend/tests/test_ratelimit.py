"""Rate limiting (Phase 2, H). Redis is exercised through an in-process fake; no live Redis was used."""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core import ratelimit
from app.core.config import Settings
from app.services.auth import get_current_user


class FakeRedis:
    def __init__(self):
        self.d, self.ttl = {}, {}

    def incr(self, k):
        self.d[k] = self.d.get(k, 0) + 1
        return self.d[k]

    def expire(self, k, s):
        self.ttl[k] = s


def _app(user_id="u1"):
    app = FastAPI()

    @app.get("/auth-ish", dependencies=[Depends(ratelimit.rate_limit_auth)])
    def a():
        return {"ok": 1}

    @app.get("/work", dependencies=[Depends(ratelimit.rate_limit_user("workflow", "workflow_rate_limit_per_minute"))])
    def w():
        return {"ok": 1}

    uid = {"v": user_id}
    app.dependency_overrides[get_current_user] = lambda: {"sub": uid["v"]}
    return app, uid


def _settings(monkeypatch, **kw):
    s = Settings(_env_file=None, **kw)
    monkeypatch.setattr(ratelimit, "get_settings", lambda: s)
    return s


def test_per_user_limit_isolated_and_retry_after(monkeypatch):
    _settings(monkeypatch, workflow_rate_limit_per_minute=2)
    app, uid = _app()
    c = TestClient(app)
    assert [c.get("/work").status_code for _ in range(3)] == [200, 200, 429]
    r = c.get("/work")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    uid["v"] = "u2"  # another user is unaffected
    assert c.get("/work").status_code == 200


def test_xff_ignored_when_no_trusted_proxy(monkeypatch):
    _settings(monkeypatch, auth_rate_limit_per_minute=2, trusted_proxy_count=0)
    c = TestClient(_app()[0])
    codes = [c.get("/auth-ish", headers={"X-Forwarded-For": f"9.9.9.{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]  # rotating a spoofed header does not evade the limit


def test_xff_uses_rightmost_trusted_hop(monkeypatch):
    _settings(monkeypatch, auth_rate_limit_per_minute=1, trusted_proxy_count=1)
    c = TestClient(_app()[0])
    # Client-supplied left entries vary, but the proxy-appended right entry is the real client.
    assert c.get("/auth-ish", headers={"X-Forwarded-For": "1.1.1.1, 5.5.5.5"}).status_code == 200
    assert c.get("/auth-ish", headers={"X-Forwarded-For": "2.2.2.2, 5.5.5.5"}).status_code == 429
    assert c.get("/auth-ish", headers={"X-Forwarded-For": "2.2.2.2, 6.6.6.6"}).status_code == 200


def test_xff_with_too_few_hops_falls_back_to_peer(monkeypatch):
    _settings(monkeypatch, auth_rate_limit_per_minute=1, trusted_proxy_count=2)
    c = TestClient(_app()[0])
    assert c.get("/auth-ish", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert c.get("/auth-ish", headers={"X-Forwarded-For": "3.3.3.3"}).status_code == 429


def test_shared_store_counts_across_instances(monkeypatch):
    _settings(monkeypatch, workflow_rate_limit_per_minute=2)
    fake = FakeRedis()
    monkeypatch.setattr(ratelimit, "_redis", lambda: fake)
    c = TestClient(_app()[0])
    assert [c.get("/work").status_code for _ in range(3)] == [200, 200, 429]
    assert fake.ttl  # window expiry was set
    ratelimit._hits.clear()  # a "second worker" has an empty local store but sees the shared counter
    assert c.get("/work").status_code == 429


def test_redis_failure_fails_open_to_memory_and_still_limits(monkeypatch, caplog):
    _settings(monkeypatch, workflow_rate_limit_per_minute=1)

    class Broken:
        def incr(self, k):
            raise ConnectionError("down")

    monkeypatch.setattr(ratelimit, "_redis", lambda: Broken())
    c = TestClient(_app()[0])
    with caplog.at_level("WARNING"):
        assert [c.get("/work").status_code for _ in range(2)] == [200, 429]
    assert any("rate_limit_redis_unavailable" in m for m in caplog.messages)


def test_zero_disables(monkeypatch):
    _settings(monkeypatch, workflow_rate_limit_per_minute=0)
    c = TestClient(_app()[0])
    assert all(c.get("/work").status_code == 200 for _ in range(10))
