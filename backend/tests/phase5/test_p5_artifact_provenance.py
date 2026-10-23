"""Phase 5 / Stage 7: a model artifact must say which data, code, protocol and configuration produced it - or declare, field by field,
that it cannot. Synthetic data only; nothing here is evidence about a real model."""
import copy
import json
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from app.main import app
from app.services.auth import get_current_user
from ml.training import train_final as TF
from ml.validation import artifact_contract as AC
from ml.validation import candidates as C
from ml.validation import data as D
from ml.validation import fingerprint as F
from tests.phase4.helpers_p4 import forged_disease_csv

MODELS = Path(F.MODELS)


def _strip_volatile(p):
    p = copy.deepcopy(p)
    p.pop("artifact_first_committed_in", None)          # depends on git history of the checkout
    return p


# ---------------------------------------------------------------------------------------------- the shipped artifacts
@pytest.mark.parametrize("kind", ["disease", "risk"])
def test_committed_sidecar_is_exactly_what_the_generator_produces_from_the_artifact(kind):
    committed = json.loads(AC.sidecar_path(kind).read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = AC.legacy_provenance(kind)
    assert _strip_volatile(committed) == _strip_volatile(fresh), "stale or hand-edited sidecar: regenerate with python -m ml.validation.artifact_contract"


@pytest.mark.parametrize("kind", ["disease", "risk"])
def test_shipped_artifacts_are_reported_incomplete_never_complete(kind):
    rep = AC.validate_provenance(json.loads(AC.sidecar_path(kind).read_text()))
    assert rep["status"] == AC.INCOMPLETE_DECLARED
    for f in ("dataset_manifest", "protocol_version", "training_code_revision", "training_date"):
        assert f in rep["unrecoverable"], f
    prov = json.loads(AC.sidecar_path(kind).read_text())
    assert prov["dataset_manifest"]["dataset_sha256"] is None            # no invented hash
    assert prov["training_date"] is None and prov["training_code_revision"] is None
    man = json.loads((MODELS / "MANIFEST.json").read_text())["sha256"]
    assert prov["artifact_checksum"]["sha256"] == man[f"{F.STEMS[kind]}.joblib"]


def test_units_are_taken_only_from_feature_names_and_never_guessed():
    u = AC.feature_units(["MDVP:Fo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer(dB)", "HNR", "RPDE", "delta_Jitter(%)", "delta_HNR", "test_time", "baseline_total_UPDRS"])
    assert u["MDVP:Fo(Hz)"]["unit"] == "Hz" and u["MDVP:Jitter(%)"]["unit"] == "percent" and u["MDVP:Shimmer(dB)"]["unit"] == "dB"
    assert u["delta_Jitter(%)"]["unit"] == "percent" and u["delta_Jitter(%)"]["is_delta_vs_first_visit"] is True
    assert u["HNR"]["unit"] is None and u["HNR"]["status"].startswith("NOT_VERIFIED") and u["RPDE"]["unit"] is None
    assert u["test_time"]["unit"] == "days" and u["test_time"]["convention"] == "days_since_recruitment_v1"


# ---------------------------------------------------------------------------------------------- the validator
def _complete():
    return AC.build_training_provenance(
        kind="risk", stem="m", model_version="1.0.0", features=["a", "b"], dataset_info={"sha256": "x" * 64, "n_rows": 10, "n_subjects": 3},
        manifest_status={"status": "PINNED_MATCH"}, protocol_version="p4a-1", seed=1, training_date="2026-01-01T00:00:00Z",
        X=pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 5.0]}), validation_pointer="results/x.json", threshold=0.5, threshold_rationale="fixed a priori",
        artifact_sha256="a" * 64, code_revision="deadbeef")


def test_complete_record_is_complete_and_any_missing_field_invalidates_it():
    p = _complete()
    assert AC.validate_provenance(p)["status"] == AC.COMPLETE
    for f in AC.REQUIRED_FIELDS:
        q = copy.deepcopy(p)
        del q[f]
        assert AC.validate_provenance(q)["status"] == AC.INVALID, f


def test_an_empty_required_field_is_invalid_unless_declared_unrecoverable_and_then_never_complete():
    p = _complete()
    p["training_date"] = None
    assert AC.validate_provenance(p)["status"] == AC.INVALID
    p["unrecoverable"] = {"training_date": "not recorded"}
    assert AC.validate_provenance(p)["status"] == AC.INCOMPLETE_DECLARED


def test_a_malformed_checksum_is_invalid():
    p = _complete()
    p["artifact_checksum"]["sha256"] = "not-a-hash"
    assert AC.validate_provenance(p)["status"] == AC.INVALID


def test_missing_sidecar_is_reported_invalid_not_raised(tmp_path):
    assert AC.load_sidecar_report("risk", tmp_path)["status"] == AC.INVALID


# ---------------------------------------------------------------------------------------------- newly trained artifacts (E05)
def _train(tmp_path, cands, seed=1):
    csv = forged_disease_csv(tmp_path / "parkinsons.data", seed=11)
    df, info = D.load_disease(csv)
    out = tmp_path / "v2"
    meta = TF.train("parkinsons_demo_classifier", "voice", df[D.DISEASE_FEATURES], df["status"], df["subject"], cands,
                    dataset_info=info, out_dir=out, seed=seed, inner_k=3, results_pointer="ml/validation/results/phase4a/disease_p4a.json",
                    manifest_status={"status": "UNPINNED"})
    jl = out / "parkinsons_demo_classifier.joblib"
    (out / "MANIFEST.json").write_text(json.dumps({"sha256": {jl.name: hashlib.sha256(jl.read_bytes()).hexdigest()}}))
    return csv, out, meta


def test_new_artifact_carries_a_complete_provenance_record(tmp_path):
    csv, out, meta = _train(tmp_path, {"LR": C._lr(1.0)})
    prov = json.loads((out / "parkinsons_demo_classifier_provenance.json").read_text())
    assert AC.validate_provenance(prov)["status"] == AC.COMPLETE and meta["provenance_status"] == AC.COMPLETE
    assert prov["dataset_manifest"]["dataset_sha256"] == D.sha256_file(csv)
    assert prov["protocol_version"] == "p4a-1" and prov["random_seeds"] == {"training_seed": 1}
    assert prov["artifact_checksum"]["sha256"] == hashlib.sha256((out / "parkinsons_demo_classifier.joblib").read_bytes()).hexdigest()
    assert prov["calibration"]["calibrated"] is False and prov["decision_threshold"]["value"] == 0.5


def test_E05_retrained_pipelines_name_their_steps_differently_and_must_still_authenticate(tmp_path):
    """Before the fix fingerprint.artifact_fingerprint looked up named_steps['scaler'] only; retrained pipelines call it 's'."""
    csv, out, _ = _train(tmp_path, {"LR": C._lr(1.0)})
    model, _ = F.load_artifact("disease", out)
    assert "scaler" not in model.named_steps and F.find_scaler(model) is model.named_steps["s"]
    assert F.authenticate("disease", csv, out)["status"] == F.AUTHENTICATED
    other = forged_disease_csv(tmp_path / "other.data", seed=12)
    assert F.authenticate("disease", other, out)["status"] == F.NOT_TRAINING_DATA


def test_E05_a_model_without_any_scaler_is_authenticated_through_the_sidecar_moments(tmp_path):
    hgb = {"HGB": lambda: Pipeline([("c", HistGradientBoostingClassifier(max_depth=2, max_iter=20, random_state=0))])}
    csv, out, _ = _train(tmp_path, hgb)
    model, _ = F.load_artifact("disease", out)
    assert F.find_scaler(model) is None
    assert F.authenticate("disease", csv, out)["status"] == F.AUTHENTICATED
    (out / "parkinsons_demo_classifier_provenance.json").unlink()                   # without the sidecar it cannot be fingerprinted: fail closed
    assert F.authenticate("disease", csv, out)["status"] == F.NOT_COMPARABLE


def test_serving_refuses_a_model_it_cannot_derive_an_input_domain_for():
    import app.api.routes.ml_demo as md
    no_scaler = Pipeline([("c", HistGradientBoostingClassifier(max_iter=5))])
    with pytest.raises(HTTPException) as e:
        md._scaler_of(no_scaler)
    assert e.value.status_code == 503


# ---------------------------------------------------------------------------------------------- API exposure
@pytest.mark.parametrize("path", ["/api/v1/demo/parkinsons-classifier/info", "/api/v1/demo/parkinsons-risk-classifier/info"])
def test_info_endpoints_expose_provenance_status(path):
    app.dependency_overrides[get_current_user] = lambda: {"sub": "u", "_access_token": "t"}
    try:
        body = TestClient(app).get(path).json()
    finally:
        app.dependency_overrides.clear()
    assert body["provenance"]["status"] == AC.INCOMPLETE_DECLARED and "training_date" in body["provenance"]["unrecoverable"]


# ---------------------------------------------------------------------------------------------- E06: safe loader vs boosting models
def test_E06_boosting_pipeline_round_trips_through_the_restricted_loader(tmp_path):
    import joblib
    from app.ml.safe_load import restricted_joblib_load
    X = np.random.default_rng(0).normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int)
    m = Pipeline([("c", HistGradientBoostingClassifier(max_iter=8, random_state=0))]).fit(X, y)
    p = tmp_path / "hgb.joblib"
    joblib.dump(m, p)
    loaded = restricted_joblib_load(p)
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X))


@pytest.mark.parametrize("module,name", [("numpy.random.bit_generator", "BitGenerator"), ("numpy.random._pcg64", "PCG64DXSM"),
                                         ("numpy", "load"), ("os", "system"), ("builtins", "eval"), ("numpy.random", "seed")])
def test_E06_the_allowlist_addition_is_exact_and_other_names_are_still_refused(module, name):
    from app.ml.safe_load import UnsafeModelError, _check
    with pytest.raises(UnsafeModelError):
        _check(module, name)


def test_E06_the_three_added_names_are_allowed():
    from app.ml.safe_load import _check
    for pair in [("numpy.random._pcg64", "PCG64"), ("numpy.random.bit_generator", "SeedSequence"),
                 ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence")]:
        _check(*pair)
