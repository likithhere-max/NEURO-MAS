"""Phase 5 / Stage 5: ENGINEERING STRESS TESTS of the data and validation pipeline.  They show how the code behaves when inputs are corrupted,
shifted or malformed. They are NOT evidence of real-world robustness, and no metric from a perturbed set is a performance estimate."""
import numpy as np
import pandas as pd
import pytest

from ml.validation import data as D
from ml.validation import exclusions as E
from ml.validation import fingerprint as F
from tests.phase4.helpers_p4 import forged_disease_csv, make_artifact_dir
from tests.phase5.helpers_p5 import forged_risk_csv


@pytest.fixture()
def disease(tmp_path):
    f = forged_disease_csv(tmp_path / "d.csv", seed=21)
    raw = pd.read_csv(f)
    return f, raw, make_artifact_dir(tmp_path / "m", "disease", raw), tmp_path


def _save(df, p):
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------------------------- corrupted / malformed files
def test_missing_values_are_refused_not_imputed(disease):
    _, raw, _, tmp = disease
    bad = raw.copy()
    bad.loc[5, "HNR"] = np.nan
    with pytest.raises(D.DatasetVerificationError, match="missing"):
        D.load_disease(_save(bad, tmp / "nan.csv"))


def test_non_finite_values_are_refused(disease):
    _, raw, _, tmp = disease
    bad = raw.copy()
    bad.loc[5, "HNR"] = np.inf
    with pytest.raises(D.DatasetVerificationError):
        D.load_disease(_save(bad, tmp / "inf.csv"))


def test_unexpected_label_category_is_refused(disease):
    _, raw, _, tmp = disease
    bad = raw.copy()
    bad.loc[0, "status"] = 2
    with pytest.raises(D.DatasetVerificationError, match="0/1"):
        D.load_disease(_save(bad, tmp / "cat.csv"))


def test_duplicated_records_change_the_counts_and_are_refused(disease):
    _, raw, _, tmp = disease
    with pytest.raises(D.DatasetVerificationError, match="expected 195"):
        D.load_disease(_save(pd.concat([raw, raw.iloc[:3]]), tmp / "dup.csv"))


def test_unparseable_recording_names_are_refused(disease):
    _, raw, _, tmp = disease
    bad = raw.copy()
    bad.loc[0, "name"] = "recording-1"
    with pytest.raises(D.DatasetVerificationError, match="unparseable"):
        D.load_disease(_save(bad, tmp / "name.csv"))


def test_subject_with_two_labels_is_refused(disease):
    _, raw, _, tmp = disease
    bad = raw.copy()
    s = bad.loc[0, "name"].rsplit("_", 1)[0]
    idx = bad.index[bad["name"].str.startswith(s)]
    bad.loc[idx[0], "status"] = 1 - bad.loc[idx[0], "status"]               # one recording of a subject now disagrees with the others
    assert bad.loc[idx, "status"].nunique() == 2
    with pytest.raises(D.DatasetVerificationError, match="mixed labels"):
        D.load_disease(_save(bad, tmp / "mix.csv"), strict_counts=False)


# ---------------------------------------------------------------------------------------------- preprocessing / unit / schema changes
@pytest.mark.parametrize("change", ["unit_x100", "column_renamed", "column_dropped", "column_swap"])
def test_a_changed_schema_or_unit_never_authenticates(disease, change):
    f, raw, adir, tmp = disease
    bad = raw.copy()
    if change == "unit_x100":
        bad["MDVP:Jitter(%)"] = bad["MDVP:Jitter(%)"] * 100          # e.g. percent vs fraction
    elif change == "column_renamed":
        bad = bad.rename(columns={"HNR": "hnr_db"})
    elif change == "column_dropped":
        bad = bad.drop(columns=["spread1"])
    else:
        bad[["HNR", "NHR"]] = bad[["NHR", "HNR"]].to_numpy()
    assert F.authenticate("disease", _save(bad, tmp / "x.csv"), adir)["status"] in (F.NOT_TRAINING_DATA, F.NOT_COMPARABLE)


def test_row_order_and_float_formatting_do_not_matter_but_a_tiny_edit_does(disease):
    f, raw, adir, tmp = disease
    shuffled = raw.sample(frac=1.0, random_state=1)
    assert F.authenticate("disease", _save(shuffled, tmp / "s.csv"), adir)["status"] == F.AUTHENTICATED
    nudged = raw.copy()
    nudged.loc[10, "HNR"] += 0.05
    assert F.authenticate("disease", _save(nudged, tmp / "n.csv"), adir)["status"] == F.NOT_TRAINING_DATA


# ---------------------------------------------------------------------------------------------- risk file: duplicate / out-of-order / shifted records
def test_risk_out_of_order_and_duplicate_records_are_handled_deterministically_and_reported(tmp_path):
    raw = pd.read_csv(forged_risk_csv(tmp_path / "r.csv", seed=31))
    clean_built, _ = D.build_risk_frame(raw)
    shuffled = raw.sample(frac=1.0, random_state=0).reset_index(drop=True)
    shuf_built, _ = D.build_risk_frame(shuffled)
    key = ["subject#", "test_time"]
    a = clean_built.sort_values(key).reset_index(drop=True)[key + ["worsening", "baseline_total_UPDRS"]]
    b = shuf_built.sort_values(key).reset_index(drop=True)[key + ["worsening", "baseline_total_UPDRS"]]
    pd.testing.assert_frame_equal(a, b)                                                    # row order in the file does not change the task
    dq = E.risk_exclusion_report(shuffled)["data_quality"]
    assert dq["subjects_with_rows_out_of_time_order_in_file"] > 30
    dup = pd.concat([raw, raw.iloc[:5]], ignore_index=True)
    assert E.risk_exclusion_report(dup)["data_quality"]["exact_duplicate_rows"] == 5
    with pytest.raises(D.DatasetVerificationError):
        D.load_risk(_save(dup, tmp_path / "dup.csv"))                                      # counts no longer match the published structure


def test_risk_temporal_shift_in_test_time_changes_the_fingerprint(tmp_path):
    f = forged_risk_csv(tmp_path / "r.csv", seed=32)
    raw = pd.read_csv(f)
    adir = make_artifact_dir(tmp_path / "m", "risk", raw)
    assert F.authenticate("risk", f, adir)["status"] == F.AUTHENTICATED
    shifted = raw.copy()
    shifted["test_time"] = shifted["test_time"] + 30.0                                     # every session 30 days later
    assert F.authenticate("risk", _save(shifted, tmp_path / "shift.csv"), adir)["status"] == F.NOT_TRAINING_DATA


def test_changed_class_prevalence_is_visible_in_the_report_not_hidden(tmp_path):
    raw = pd.read_csv(forged_risk_csv(tmp_path / "r.csv", seed=33))
    low = raw.copy()
    low["total_UPDRS"] = low.groupby("subject#")["total_UPDRS"].transform("first")         # nobody worsens
    built, _ = D.build_risk_frame(low)
    assert built["worsening"].mean() == 0.0                                                # a degenerate prevalence the metrics code must meet honestly
    from ml.evalkit import score_metrics
    m = score_metrics(built["worsening"], np.full(len(built), 0.3))
    assert m["roc_auc"] is None and "only one class" in m["undefined"]["roc_auc"]          # undefined, not 0.5 and not an error
