"""Tests for ml/validation: dataset verification (fail loudly), result envelope honesty, artifacts check.
Synthetic fixtures test CODE PATHS only; nothing here is evidence about the real models."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.validation import data as D
from ml.validation import validate_models as V


def test_synthetic_disease_frame_passes_structure_checks_but_not_publisher_counts():
    df, info = V.synthetic_disease(0)
    assert info["n_subjects"] == 24 and info["subjects_pd"] == 18
    with pytest.raises(D.DatasetVerificationError, match="expected 195 rows"):
        D.verify_disease_frame(df.drop(columns="subject"), strict_counts=True)


def test_disease_verification_rejects_mixed_label_subject_missing_columns_and_nan():
    df, _ = V.synthetic_disease(0)
    base = df.drop(columns="subject")
    bad = base.copy()
    bad.loc[0, "status"] = 1 - bad.loc[0, "status"]
    with pytest.raises(D.DatasetVerificationError, match="mixed labels"):
        D.verify_disease_frame(bad, strict_counts=False)
    with pytest.raises(D.DatasetVerificationError, match="missing columns"):
        D.verify_disease_frame(base.drop(columns=["HNR"]), strict_counts=False)
    nan = base.copy()
    nan.loc[3, "HNR"] = np.nan
    with pytest.raises(D.DatasetVerificationError, match="missing values"):
        D.verify_disease_frame(nan, strict_counts=False)
    badname = base.copy()
    badname.loc[0, "name"] = "oops"
    with pytest.raises(D.DatasetVerificationError, match="unparseable"):
        D.verify_disease_frame(badname, strict_counts=False)


def test_risk_frame_label_definition_matches_spec_on_a_hand_example():
    raw = pd.DataFrame({"subject#": [1, 1, 1, 2, 2], "age": 60, "sex": 0, "test_time": [0, 10, 20, 0, 10],
                        "motor_UPDRS": 0, "total_UPDRS": [20.0, 22.9, 23.0, 30.0, 25.0], "HNR": [20.0, 21.0, 19.0, 18.0, 18.0]})
    df, delta = D.build_risk_frame(raw, threshold=3.0)
    assert delta == ["delta_HNR"]
    # first visits are dropped; label = (total(t) - total(first)) >= 3.0
    assert df["worsening"].tolist() == [0, 1, 0]       # 2.9 -> 0 ; 3.0 -> 1 (boundary inclusive) ; -5 -> 0
    assert df["delta_HNR"].tolist() == [1.0, -1.0, 0.0]
    assert df["baseline_total_UPDRS"].tolist() == [20.0, 20.0, 30.0]


def test_risk_frame_does_not_use_future_visits_for_baseline():
    # unsorted input must still use the EARLIEST visit as baseline
    raw = pd.DataFrame({"subject#": [1, 1, 1], "age": 60, "sex": 0, "test_time": [20, 0, 10], "motor_UPDRS": 0,
                        "total_UPDRS": [25.0, 20.0, 21.0], "HNR": [1.0, 2.0, 3.0]})
    df, _ = D.build_risk_frame(raw)
    assert df["baseline_total_UPDRS"].unique().tolist() == [20.0]


def test_lock_file_blocks_a_modified_dataset(tmp_path, monkeypatch):
    f = tmp_path / "parkinsons.data"
    f.write_text("a,b\n1,2\n")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"parkinsons": {"sha256": "0" * 64}}))
    monkeypatch.setattr(D, "LOCK", lock)
    with pytest.raises(D.DatasetVerificationError, match="differs"):
        D.check_lock("parkinsons", f)


def test_results_envelope_never_claims_clinical_validation_and_flags_synthetic_runs():
    df, info = V.synthetic_disease(1, n_subjects=8, rec=3)
    r = V.run_disease_df(df, info, seed=1, n_boot=25, n_perm=3, fast=True, evidence="SYNTHETIC_FIXTURE_NOT_EVIDENCE")
    assert r["clinically_validated"] is False and r["status"] == "completed"
    assert r["evidence_status"] == "SYNTHETIC_FIXTURE_NOT_EVIDENCE" and r["dataset"]["SYNTHETIC"] is True
    assert set(r["nested_performance"]) >= {"recording_level", "subject_level", "ppv_npv_at_prevalence"}
    assert "optimistic" in r["biased_protocol_for_comparison"]["note"]
    assert r["nested_performance"]["subject_level"]["level"] == "subject"


def test_failed_run_is_reported_as_failed_not_partial(tmp_path):
    rc = V.main(["disease", "--csv", str(tmp_path / "missing.csv"), "--out", str(tmp_path)])
    assert rc == 1
    out = json.loads((tmp_path / "disease_FAILED.json").read_text())
    assert out["status"] == "failed" and out["evidence_status"] == "NONE" and out["clinically_validated"] is False


def test_artifacts_check_runs_on_repo_files_and_matches_manifest():
    r = V.run_artifacts()
    assert r["status"] == "completed" and r["evidence_status"] == "REAL_RUN_ON_REPO_FILES"
    for stem, a in r["artifacts"].items():
        assert a["manifest_match"], stem
        assert a["feature_count_matches_metadata"], stem


def test_train_final_writes_provenance_domain_and_no_performance_claims(tmp_path):
    from ml.training.train_final import train
    from ml.validation import candidates as C
    df, info = V.synthetic_disease(2, n_subjects=10, rec=4)
    meta = train("m", "voice", df[D.DISEASE_FEATURES], df["status"], df["subject"], {"LR": C.disease_candidates(True)["LR_C1"]},
                 dataset_info=info, out_dir=tmp_path, seed=2, inner_k=3)
    assert (tmp_path / "m.joblib").exists() and (tmp_path / "m_domain.json").exists()
    saved = json.loads((tmp_path / "m_metadata.json").read_text())
    assert saved["sklearn_version"] and saved["feature_columns_in_order"] == D.DISEASE_FEATURES
    assert saved["validation"]["clinically_validated"] is False and saved["validation"]["performance_numbers_in_this_file"].startswith("none")
    assert not {"accuracy", "roc_auc", "f1", "loso_metrics"} & set(saved)
    from app.ml.ood import InputDomain
    assert InputDomain.from_json((tmp_path / "m_domain.json").read_text()).source == "fitted"
    assert meta["selection"]["status"] in ("selected", "fallback_no_valid_inner_score")
