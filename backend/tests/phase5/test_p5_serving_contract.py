"""Phase 5 / Stage 2 + 7: one convention for test_time, identical features in training and serving, and safe handling of bad input.
Synthetic inputs only; the shipped artifact is used as a loadable object, not as evidence of anything."""
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml import contract as K
from app.schemas.ml_demo import ParkinsonsRiskFeaturesIn
from app.services.auth import get_current_user
from ml.validation import data as D

BACKEND = Path(__file__).resolve().parents[2]
client = TestClient(app)
RISK = "/api/v1/demo/parkinsons-risk-classifier"
VOICE = "/api/v1/demo/parkinsons-classifier"


@pytest.fixture(autouse=True)
def _user():
    app.dependency_overrides[get_current_user] = lambda: {"sub": "u", "_access_token": "t"}
    yield
    app.dependency_overrides.clear()


def _meta(stem):
    return json.loads((BACKEND / f"ml/models/{stem}_metadata.json").read_text())


def _mean_row(stem):
    import joblib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = joblib.load(BACKEND / f"ml/models/{stem}.joblib").named_steps["scaler"]
    return dict(zip(_meta(stem)["feature_columns_in_order"], map(float, sc.mean_)))


# ---------------------------------------------------------------------------------------------- one convention, everywhere
def test_schema_and_openapi_use_the_single_convention():
    desc = ParkinsonsRiskFeaturesIn.model_fields["test_time"].description
    assert desc == K.TEST_TIME_DESCRIPTION and "RECRUITMENT" in desc and "NOT days since the baseline visit" in desc
    spec = client.get("/openapi.json").json()
    props = spec["components"]["schemas"]["ParkinsonsRiskFeaturesIn"]["properties"]["test_time"]
    assert props["description"] == K.TEST_TIME_DESCRIPTION


def test_no_serving_or_ml_module_still_calls_test_time_days_since_baseline():
    """The Phase 4A schema fix left app/ml/ood.py saying 'days since baseline visit'. Scan so it cannot come back."""
    offenders = []
    for root in ("app", "ml"):
        for p in (BACKEND / root).rglob("*.py"):
            if "contract.py" in p.name or "tests" in p.parts:
                continue
            for i, line in enumerate(p.read_text().splitlines(), 1):
                low = line.lower()
                if ("test_time" in low or "elapsed" in low) and re.search(r"days since (the )?baseline", low) and " not " not in f" {low} ":
                    offenders.append(f"{p.relative_to(BACKEND)}:{i}")
    assert offenders == []
    ood = (BACKEND / "app/ml/ood.py").read_text()
    assert "days since baseline visit cannot be negative" not in ood and "RECRUITMENT" in ood


def test_the_convention_id_is_recorded_in_the_artifact_provenance():
    prov = json.loads((BACKEND / "ml/models/parkinsons_risk_classifier_provenance.json").read_text())
    assert prov["feature_units"]["test_time"]["convention"] == K.TEST_TIME_CONVENTION_ID == "days_since_recruitment_v1"
    assert K.TEST_TIME_MISMATCH_NOTE in prov["known_limitations"]


# ---------------------------------------------------------------------------------------------- training / serving feature parity
def _raw_with_all_voice_columns(seed=0, n_sub=3, visits=6):
    voice = [c[len("delta_"):] for c in _meta("parkinsons_risk_classifier")["feature_columns_in_order"] if c.startswith("delta_")]
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_sub + 1):
        for i, t in enumerate(np.sort(rng.uniform(1, 170, visits))):
            rows.append({"subject#": s, "age": 60, "sex": 0, "test_time": float(t), "motor_UPDRS": 10.0, "total_UPDRS": 20.0 + 0.05 * t,
                         **{c: float(rng.normal(0, 1)) for c in voice}})
    return pd.DataFrame(rows), voice


def test_every_training_row_maps_to_the_identical_vector_through_the_serving_path():
    """Training side: build_risk_frame. Serving side: the request schema (by alias) -> DataFrame in metadata column order -> model input.
    Exact float equality for all 18 features, for every row; and the feature order equals the shipped metadata's."""
    raw, voice = _raw_with_all_voice_columns()
    built, dcols = D.build_risk_frame(raw)
    order = _meta("parkinsons_risk_classifier")["feature_columns_in_order"]
    assert dcols + ["test_time", "baseline_total_UPDRS"] == order
    for _, r in built.iterrows():
        payload = ParkinsonsRiskFeaturesIn(**{c: float(r[c]) for c in order}).model_dump(by_alias=True)
        served = pd.DataFrame([payload])[order]
        assert served.iloc[0].to_numpy(float).tolist() == [float(r[c]) for c in order]
    assert len(order) == 18 and not (set(order) - set(ParkinsonsRiskFeaturesIn.model_json_schema(by_alias=True)["properties"]))


def test_serving_probability_equals_direct_model_probability_for_the_same_valid_input():
    import joblib
    mean = _mean_row("parkinsons_risk_classifier")
    order = _meta("parkinsons_risk_classifier")["feature_columns_in_order"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(BACKEND / "ml/models/parkinsons_risk_classifier.joblib")
    body = client.post(RISK, json=mean).json()
    direct = model.predict_proba(pd.DataFrame([mean])[order])[0]
    assert body["abstained"] is False
    assert body["probability_worsening"] == round(float(direct[1]), 4) and body["probability_stable_or_improving"] == round(float(direct[0]), 4)
    assert abs(body["probability_worsening"] + body["probability_stable_or_improving"] - 1) < 2e-4         # bounds and class ordering
    assert body["predicted_class"] == int(direct[1] >= direct[0]) and body["probability_calibrated"] is False


# ---------------------------------------------------------------------------------------------- invalid input never yields a confident number
def test_missing_or_mistyped_fields_are_rejected_not_defaulted():
    mean = _mean_row("parkinsons_risk_classifier")
    for bad in ({k: v for k, v in mean.items() if k != "test_time"}, {**mean, "test_time": "soon"}, {**mean, "baseline_total_UPDRS": None}, {}):
        r = client.post(RISK, json=bad)
        assert r.status_code == 422, bad


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_never_produce_a_prediction(literal):
    mean = _mean_row("parkinsons_risk_classifier")
    body = json.dumps({**mean, "test_time": 1.0}).replace('"test_time": 1.0', f'"test_time": {literal}')
    r = client.post(RISK, content=body, headers={"content-type": "application/json"})
    if r.status_code == 200:
        j = r.json()
        assert j["abstained"] is True and j["probability_worsening"] is None and j["predicted_class"] is None
    else:
        assert r.status_code == 422


@pytest.mark.parametrize("field,value", [("test_time", 1e9), ("test_time", -0.001), ("baseline_total_UPDRS", 1e6), ("baseline_total_UPDRS", -5.0),
                                         ("delta_Jitter(%)", 1e6), ("delta_HNR", -1e6)])
def test_out_of_range_values_abstain(field, value):
    mean = _mean_row("parkinsons_risk_classifier")
    j = client.post(RISK, json={**mean, field: value}).json()
    assert j["abstained"] is True and j["probability_worsening"] is None and j["domain"]["in_domain"] is False


def test_voice_endpoint_abstains_on_physically_impossible_input():
    row = _mean_row("parkinsons_demo_classifier")
    j = client.post(VOICE, json={**row, "MDVP:Fo(Hz)": -50.0}).json()
    assert j["abstained"] is True and j.get("probability_parkinsons") is None


def test_model_unavailable_is_a_503_not_a_default_answer(monkeypatch):
    import app.api.routes.ml_demo as md
    monkeypatch.setattr(md, "_risk_model", None)
    monkeypatch.setattr(md, "RISK_MODEL_PATH", BACKEND / "ml/models/does_not_exist.joblib")
    r = client.post(RISK, json=_mean_row("parkinsons_risk_classifier"))
    assert r.status_code == 503 and "probability" not in r.text.lower().replace("probability_", "")


def test_duplicate_requests_are_deterministic():
    mean = _mean_row("parkinsons_risk_classifier")
    a, b = client.post(RISK, json=mean).json(), client.post(RISK, json=mean).json()
    assert a == b
