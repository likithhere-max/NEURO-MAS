import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services.auth import get_current_user

client = TestClient(app)
META = json.loads((Path(__file__).resolve().parent.parent / "ml/models/parkinsons_demo_classifier_metadata.json").read_text())
ROW = {c: 1.0 for c in META["feature_columns_in_order"]}


@pytest.fixture(autouse=True)
def _user():
    app.dependency_overrides[get_current_user] = lambda: {"sub": "u", "_access_token": "t"}
    yield
    app.dependency_overrides.clear()


def test_ml_demo_requires_auth():
    app.dependency_overrides.clear()
    assert client.post("/api/v1/demo/parkinsons-classifier", json=ROW).status_code in (401, 403)


def test_ml_demo_accepts_finite_row_and_returns_probabilities():
    r = client.post("/api/v1/demo/parkinsons-classifier", json=ROW)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability_parkinsons"] <= 1 and body["probability_healthy"] == pytest.approx(1 - body["probability_parkinsons"], abs=1e-3)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_ml_demo_rejects_non_finite(bad):
    raw = json.dumps(ROW).replace("1.0", bad, 1)
    r = client.post("/api/v1/demo/parkinsons-classifier", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_ml_demo_rejects_missing_feature():
    row = dict(ROW); row.pop(next(iter(row)))
    assert client.post("/api/v1/demo/parkinsons-classifier", json=row).status_code == 422


def test_ml_demo_metadata_endpoint_discloses_limitations():
    body = client.get("/api/v1/demo/parkinsons-classifier/info").json()
    assert "disclaimer" in json.dumps(body).lower()


# ---- clinician_id resolution ----
import app.api.routes.patients as pr


class _AdminFake:
    def __init__(self, rows): self.rows = rows
    def table(self, n): return self
    def select(self, *a): return self
    def eq(self, *a): return self
    def execute(self):
        class R: pass
        r = R(); r.data = self.rows; return r


def test_clinician_can_only_assign_to_self():
    assert pr._resolve_clinician_id(None, {"sub": "doc"}, "clinician") == "doc"
    assert pr._resolve_clinician_id("doc", {"sub": "doc"}, "clinician") == "doc"
    with pytest.raises(HTTPException) as e:
        pr._resolve_clinician_id("other-doc", {"sub": "doc"}, "clinician")
    assert e.value.status_code == 403


def test_patient_may_not_name_a_non_clinician_or_garbage(monkeypatch):
    good = "11111111-1111-1111-1111-111111111111"
    monkeypatch.setattr(pr, "get_supabase_admin_client", lambda: _AdminFake([{"role": "patient"}]))
    with pytest.raises(HTTPException) as e:
        pr._resolve_clinician_id(good, {"sub": "p"}, "patient")
    assert e.value.status_code == 422
    with pytest.raises(HTTPException):
        pr._resolve_clinician_id("not-a-uuid", {"sub": "p"}, "patient")
    monkeypatch.setattr(pr, "get_supabase_admin_client", lambda: _AdminFake([{"role": "clinician"}]))
    assert pr._resolve_clinician_id(good, {"sub": "p"}, "patient") == good
    assert pr._resolve_clinician_id(None, {"sub": "p"}, "patient") is None


def test_validation_errors_do_not_echo_the_rejected_input():
    secret = "SECRET-PATIENT-TEXT-" + "x" * 10
    r = client.post("/api/v1/patients", json={"full_name": secret, "age": 999})
    assert r.status_code == 422
    assert secret not in r.text and set(r.json()["detail"][0]) == {"loc", "msg", "type"}


def test_tampered_model_artifact_is_refused(monkeypatch, tmp_path):
    import app.api.routes.ml_demo as md
    fake_dir = tmp_path
    (fake_dir / "m.joblib").write_bytes(b"not a real pickle")
    (fake_dir / "MANIFEST.json").write_text(json.dumps({"sha256": {"m.joblib": "0" * 64}}))
    monkeypatch.setattr(md, "MODEL_DIR", fake_dir)
    with pytest.raises(HTTPException) as e:
        md._verified_load(fake_dir / "m.joblib")
    assert e.value.status_code == 503


def test_missing_manifest_entry_is_refused(monkeypatch, tmp_path):
    import app.api.routes.ml_demo as md
    (tmp_path / "m.joblib").write_bytes(b"x")
    monkeypatch.setattr(md, "MODEL_DIR", tmp_path)
    with pytest.raises(HTTPException):
        md._verified_load(tmp_path / "m.joblib")


def test_committed_artifacts_match_manifest():
    import hashlib
    import app.api.routes.ml_demo as md
    man = json.loads((md.MODEL_DIR / "MANIFEST.json").read_text())["sha256"]
    for name, h in man.items():
        assert hashlib.sha256((md.MODEL_DIR / name).read_bytes()).hexdigest() == h
    assert set(man) == {p.name for p in md.MODEL_DIR.glob("*.joblib")}
