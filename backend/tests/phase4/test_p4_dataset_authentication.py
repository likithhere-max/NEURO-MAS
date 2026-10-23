"""Phase 4A: a data file may only be labelled as the real training data if it reproduces the shipped artifact's fingerprint."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.validation import fingerprint as F
from ml.validation import validate_models as V
from tests.phase4.helpers_p4 import forged_disease_csv, make_artifact_dir, raw_risk_frame


# ---------------------------------------------------------------------------------------------- the defect (E03)
def test_E03_forged_but_structurally_valid_file_is_not_labelled_real_data(tmp_path, monkeypatch):
    """Before the fix the CLI wrote evidence_status == 'REAL_DATA_RUN' for ANY csv of the right shape."""
    forged = forged_disease_csv(tmp_path / "parkinsons.data")
    monkeypatch.setattr(V, "run_disease_df", lambda df, info, **kw: V.envelope("disease_voice_classifier", kw["seed"], kw["evidence"]))
    out = tmp_path / "out"
    assert V.main(["disease", "--csv", str(forged), "--out", str(out)]) == 0
    payload = json.loads((out / "disease.json").read_text())
    assert payload["evidence_status"] != "REAL_DATA_RUN"
    assert payload["evidence_status"] == F.EVIDENCE_UNAUTHENTICATED
    assert payload["clinically_validated"] is False


def test_E03_only_the_authentic_matrix_earns_the_authenticated_label(tmp_path):
    forged = forged_disease_csv(tmp_path / "parkinsons.data", seed=3)
    adir = make_artifact_dir(tmp_path / "models", "disease", pd.read_csv(forged))   # stands in for the shipped artifact
    auth = F.authenticate("disease", forged, adir)
    assert auth["status"] == F.AUTHENTICATED and F.evidence_status(auth) == F.EVIDENCE_AUTHENTICATED
    other = forged_disease_csv(tmp_path / "other.data", seed=4)                    # same shape, different numbers
    auth2 = F.authenticate("disease", other, adir)
    assert auth2["status"] == F.NOT_TRAINING_DATA and F.evidence_status(auth2) == F.EVIDENCE_UNAUTHENTICATED


# ---------------------------------------------------------------------------------------------- fingerprint behaviour
def test_fingerprint_matches_itself_and_is_row_order_invariant(tmp_path):
    forged = forged_disease_csv(tmp_path / "d.csv", seed=5)
    adir = make_artifact_dir(tmp_path / "m", "disease", pd.read_csv(forged))
    assert F.authenticate("disease", forged, adir)["status"] == F.AUTHENTICATED
    shuffled = pd.read_csv(forged).sample(frac=1.0, random_state=0)
    shuffled.to_csv(tmp_path / "s.csv", index=False)
    assert F.authenticate("disease", tmp_path / "s.csv", adir)["status"] == F.AUTHENTICATED


@pytest.mark.parametrize("mutation", ["one_value_nudged", "row_dropped", "different_random_data", "rounded_to_3dp"])
def test_fingerprint_rejects_altered_or_different_data(tmp_path, mutation):
    forged = forged_disease_csv(tmp_path / "d.csv", seed=7)
    raw = pd.read_csv(forged)
    adir = make_artifact_dir(tmp_path / "m", "disease", raw)
    bad = raw.copy()
    feat = "HNR"
    if mutation == "one_value_nudged":
        bad.loc[10, feat] += 0.05          # a 0.05-SD change in ONE cell of 4,290
    elif mutation == "row_dropped":
        bad = bad.drop(index=3)
    elif mutation == "different_random_data":
        bad = pd.read_csv(forged_disease_csv(tmp_path / "other.csv", seed=99))
    elif mutation == "rounded_to_3dp":
        num = bad.columns.difference(["name", "status"])
        bad[num] = bad[num].round(3)
    bad.to_csv(tmp_path / "bad.csv", index=False)
    assert F.authenticate("disease", tmp_path / "bad.csv", adir)["status"] == F.NOT_TRAINING_DATA


def test_fingerprint_for_the_risk_task_uses_the_project_feature_construction(tmp_path):
    raw = raw_risk_frame()
    raw.to_csv(tmp_path / "r.csv", index=False)
    adir = make_artifact_dir(tmp_path / "m", "risk", raw)
    assert F.authenticate("risk", tmp_path / "r.csv", adir)["status"] == F.AUTHENTICATED
    # the label threshold is NOT part of the fingerprint, but dropping a subject's rows is
    raw[raw["subject#"] != 3].to_csv(tmp_path / "r2.csv", index=False)
    assert F.authenticate("risk", tmp_path / "r2.csv", adir)["status"] == F.NOT_TRAINING_DATA


def test_authentication_fails_closed_on_every_kind_of_bad_input(tmp_path):
    forged = forged_disease_csv(tmp_path / "d.csv")
    adir = make_artifact_dir(tmp_path / "m", "disease", pd.read_csv(forged))
    assert F.authenticate("disease", tmp_path / "missing.csv", adir)["status"] == F.NOT_COMPARABLE
    (tmp_path / "junk.csv").write_text("a,b\n1,2\n")
    assert F.authenticate("disease", tmp_path / "junk.csv", adir)["status"] == F.NOT_COMPARABLE
    # tampered artifact: the manifest hash no longer matches -> refuse to load, never "authenticated"
    (adir / "parkinsons_demo_classifier.joblib").write_bytes((adir / "parkinsons_demo_classifier.joblib").read_bytes() + b"x")
    r = F.authenticate("disease", forged, adir)
    assert r["status"] == F.NOT_COMPARABLE and "SHA-256" in r["reason"]
    assert F.evidence_status(r) == F.EVIDENCE_UNAUTHENTICATED


def test_the_shipped_artifacts_expose_the_row_counts_the_metadata_claims():
    """Independent of any data: the scalers inside the committed artifacts were fitted on exactly these many rows."""
    d, r = F.artifact_fingerprint("disease"), F.artifact_fingerprint("risk")
    assert d["n"] == 195 and len(d["features"]) == 22
    assert r["n"] == 5801 and len(r["features"]) == 18
