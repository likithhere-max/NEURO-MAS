"""Phase 4A: integrity of the pre-registered protocol runner -- freezing, fail-clearly behaviour, no substitution of data,
no patient straddling an outer boundary, no leakage, deterministic output.  Synthetic fixtures are used ONLY to exercise code paths and
to build adversarial traps; no assertion here is a claim about a real model."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ml.evalkit import nested_group_cv
from ml.validation import candidates as C
from ml.validation import data as D
from ml.validation import fingerprint as F
from ml.validation import phase4a as P
from ml.validation import phase4a_selftest as ST
from tests.phase4.helpers_p4 import forged_disease_csv


@pytest.fixture()
def lr_only(monkeypatch):
    """Fast candidate sets: the integrity properties below do not depend on which learners are compared."""
    monkeypatch.setattr(P.C, "risk_candidates", lambda fast=False: ST.lr_cands())
    monkeypatch.setattr(P.C, "disease_candidates", lambda fast=False: ST.lr_cands())


# ---------------------------------------------------------------------------------------------- protocol is frozen
def test_protocol_hash_matches_its_content_and_changes_if_any_parameter_changes():
    assert P.PROTOCOL_SHA256 == hashlib.sha256(json.dumps(P.PROTOCOL, sort_keys=True).encode()).hexdigest()
    altered = json.loads(json.dumps(P.PROTOCOL))
    altered["threshold"] = 0.4
    assert hashlib.sha256(json.dumps(altered, sort_keys=True).encode()).hexdigest() != P.PROTOCOL_SHA256


def test_protocol_pins_the_choices_that_could_be_tuned_to_flatter_the_result():
    assert P.PROTOCOL["threshold"] == 0.5 and P.PROTOCOL["selection_metric"] == "roc_auc"
    assert P.PROTOCOL["risk"]["worsening_threshold_points"] == D.WORSENING_THRESHOLD == 3.0
    assert {"TIME_BASELINE", "TIME_ONLY", "BASELINE_ONLY", "DEMOGRAPHIC_ONLY"} <= set(P.PROTOCOL["risk"]["feature_sets"])
    assert P.PROTOCOL["disease"]["baselines"] == ["dummy_prior", "classic3_LR"]


def test_every_envelope_carries_the_protocol_hash_and_never_claims_clinical_validity():
    e = P.envelope("x", "completed", "SOMETHING")
    assert e["protocol_sha256"] == P.PROTOCOL_SHA256 and e["clinically_validated"] is False


# ---------------------------------------------------------------------------------------------- fail clearly, never substitute
def test_missing_data_is_reported_as_not_run_with_exit_code_2_and_no_result_file(tmp_path, monkeypatch):
    for v in P.ENV_VARS.values():
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(P.D, "CACHE", tmp_path / "empty_cache")
    for kind in ("disease", "risk"):
        assert P.main(["run", "--kind", kind, "--csv", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "o")]) == 2
        rep = json.loads((tmp_path / "o" / f"{kind}_NOT_RUN.json").read_text())
        assert rep["status"] == "not_run" and rep["evidence_status"] == "NONE" and "No synthetic or substitute data" in rep["reason"]
        assert not (tmp_path / "o" / f"{kind}_p4a.json").exists()


def test_data_path_resolution_order_and_environment_variable(tmp_path, monkeypatch):
    f = tmp_path / "x.csv"
    f.write_text("a\n1\n")
    monkeypatch.setattr(P.D, "CACHE", tmp_path / "none")
    monkeypatch.setenv(P.ENV_VARS["risk"], str(f))
    assert P.resolve_data_path("risk") == f
    with pytest.raises(P.DataUnavailable):
        P.resolve_data_path("disease")


def test_a_structurally_valid_forged_file_runs_but_is_labelled_unauthenticated(tmp_path, monkeypatch, lr_only):
    forged = forged_disease_csv(tmp_path / "parkinsons.data")
    monkeypatch.setattr(P, "run_disease", lambda df, info, **kw: P.envelope("disease_voice_classifier", "completed", kw["evidence"]))
    assert P.main(["run", "--kind", "disease", "--csv", str(forged), "--out", str(tmp_path / "o")]) == 0
    out = json.loads((tmp_path / "o" / "disease_p4a.json").read_text())
    assert out["evidence_status"] == F.EVIDENCE_UNAUTHENTICATED and out["authentication"]["status"] == F.NOT_TRAINING_DATA


def test_fast_runs_are_marked_as_not_the_registered_protocol(tmp_path, monkeypatch):
    forged = forged_disease_csv(tmp_path / "parkinsons.data")
    monkeypatch.setattr(P, "run_disease", lambda df, info, **kw: P.envelope("d", "completed", kw["evidence"]))
    P.main(["run", "--kind", "disease", "--csv", str(forged), "--fast", "--out", str(tmp_path / "o")])
    assert json.loads((tmp_path / "o" / "disease_p4a.json").read_text())["evidence_status"].endswith("+FAST_NOT_REGISTERED_PROTOCOL")


def test_a_crash_is_reported_as_failed_not_as_a_partial_result(tmp_path, monkeypatch):
    forged = forged_disease_csv(tmp_path / "parkinsons.data")

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(P, "run_disease", boom)
    assert P.main(["run", "--kind", "disease", "--csv", str(forged), "--out", str(tmp_path / "o")]) == 1
    rep = json.loads((tmp_path / "o" / "disease_FAILED.json").read_text())
    assert rep["status"] == "failed" and rep["evidence_status"] == "NONE" and not (tmp_path / "o" / "disease_p4a.json").exists()


# ---------------------------------------------------------------------------------------------- candidates carry no hidden state
@pytest.mark.parametrize("make", [lambda: C.disease_candidates(True), lambda: C.disease_baselines(True), lambda: C.risk_candidates(True)])
def test_candidate_factories_return_fresh_unfitted_estimators_each_call(make):
    from sklearn.exceptions import NotFittedError
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted
    for name, fac in make().items():
        a, b = fac(), fac()
        assert a is not b, name                                   # a shared instance would carry fitted state between folds
        for est in (a, b):
            with pytest.raises(NotFittedError):
                check_is_fitted(est.steps[-1][1] if isinstance(est, Pipeline) else est)
            if isinstance(est, Pipeline):                         # every data-dependent step lives INSIDE the pipeline
                for _, step in est.steps[:-1]:
                    if hasattr(step, "fit"):
                        with pytest.raises(NotFittedError):
                            check_is_fitted(step)


# ---------------------------------------------------------------------------------------------- the runner on adversarial data
def _strip(d):
    d = json.loads(json.dumps(d, default=str))
    d.pop("runtime_seconds", None)
    return d


def test_risk_runner_every_patient_is_in_exactly_one_outer_test_fold_and_never_in_its_own_training_set(lr_only, monkeypatch):
    df, dcols, info = P.synthetic_risk(11)
    seen = []
    real = P.nested_group_cv

    def spy(X, y, groups, cands, **kw):
        r = real(X, y, groups, cands, **kw)
        seen.append((pd.Series(np.asarray(groups)), r))
        return r
    monkeypatch.setattr(P, "nested_group_cv", spy)
    P.run_risk(df, dcols, info, seeds=[1, 2], n_boot=20, fast=True, evidence="SYNTHETIC_FIXTURE_NOT_EVIDENCE")
    assert len(seen) == 2 * 8                                      # 8 feature sets x 2 seeds
    for g, res in seen:
        counts = pd.Series([grp for f in res.folds for grp in f.test_groups]).value_counts()
        assert set(counts.index) == set(g.unique()) and (counts == 1).all()
        for f in res.folds:
            assert f.n_train + f.n_test == len(g) and f.n_test > 0


def test_risk_runner_structure_baselines_and_dispersion(lr_only):
    df, dcols, info = P.synthetic_risk(12)
    out = P.run_risk(df, dcols, info, seeds=[1, 2], n_boot=20, fast=True, evidence="SYNTHETIC_FIXTURE_NOT_EVIDENCE")
    assert out["clinically_validated"] is False and out["evidence_status"] == "SYNTHETIC_FIXTURE_NOT_EVIDENCE"
    assert set(out["results_by_seed"][1]) >= set(P.PROTOCOL["risk"]["feature_sets"])           # incl. demographic-only baselines
    first = out["results_by_seed"][1]["FULL"]
    assert len(first["per_outer_fold_auc"]) == 10 and "pr_auc_cluster_bootstrap_ci95" in first["pooled"]
    conf = first["pooled"]["at_threshold"]["confusion"]
    assert conf["tp"] + conf["fn"] + conf["fp"] + conf["tn"] == len(df) and conf["tp"] + conf["fn"] == int(df["worsening"].sum())
    assert set(out["dispersion_across_seeds"]["FULL"]) == {"auc_by_seed", "mean", "sd", "min", "max"}
    assert len(out["dispersion_across_seeds"]["FULL"]["auc_by_seed"]) == 2
    assert "registered_rule_met" in out["voice_incremental_value"]


def test_risk_runner_is_deterministic_for_fixed_seeds(lr_only):
    df, dcols, info = P.synthetic_risk(13)
    a = _strip(P.run_risk(df, dcols, info, seeds=[5], n_boot=20, fast=True, evidence="E"))
    b = _strip(P.run_risk(df, dcols, info, seeds=[5], n_boot=20, fast=True, evidence="E"))
    assert a == b


def test_risk_runner_refuses_when_a_required_column_is_missing(lr_only):
    df, dcols, info = P.synthetic_risk(14)
    with pytest.raises(D.DatasetVerificationError):
        P.run_risk(df.drop(columns=["age"]), dcols, info, seeds=[1], n_boot=20, fast=True, evidence="E")


def test_disease_runner_structure_subject_counts_and_exact_intervals(lr_only):
    df, info = P.synthetic_disease(21, n_subjects=16, rec=4)
    out = P.run_disease(df, info, seeds=[1], n_boot=20, n_perm=5, fast=True, evidence="SYNTHETIC_FIXTURE_NOT_EVIDENCE")
    sl = out["results_by_seed"][1]["subject_level"]
    assert sl["healthy_subjects_total"] == 4 and sl["pd_subjects_total"] == 12                 # 16 subjects, 75% PD
    assert sl["specificity_clopper_pearson95"] == pytest.approx(P.clopper_pearson(sl["healthy_subjects_correct"], 4))
    assert set(out["results_by_seed"][1]["baselines"]) == {"dummy_prior", "classic3_LR"}
    assert len(out["results_by_seed"][1]["per_subject"]) == 16
    assert out["ppv_npv_at_prevalence"] is not None and out["clinically_validated"] is False


def test_disease_runner_dummy_baseline_is_exactly_chance(lr_only):
    df, info = P.synthetic_disease(22, n_subjects=16, rec=4)
    out = P.run_disease(df, info, seeds=[1], n_boot=20, n_perm=5, fast=True, evidence="E")
    b = out["results_by_seed"][1]["baselines"]["dummy_prior"]["recording_level"]
    assert b["roc_auc"] == pytest.approx(0.5)                                              # a constant score has AUC 1/2 by definition
    assert b["pr_auc"] == pytest.approx(0.75)                                              # PR-AUC of a no-skill score equals prevalence
    assert b["at_threshold"]["accuracy"] == pytest.approx(12 / 16, abs=1e-12)              # predicts the majority class (PD, 75%)


def test_dummy_prior_leave_one_subject_out_artifact_is_real_and_is_kept_but_labelled(lr_only):
    """E04b: the raw out-of-fold AUC of a constant-prior model is ~0 (each held-out subject's own label lowers the prior it is scored
    with). It must not be presented as the no-skill baseline, but it must not be hidden either."""
    df, info = P.synthetic_disease(22, n_subjects=16, rec=4)
    X, y, g = df[D.DISEASE_FEATURES].reset_index(drop=True), df["status"].reset_index(drop=True), df["subject"].reset_index(drop=True)
    r = nested_group_cv(X, y, g, {"dummy_prior": C.disease_baselines(True)["dummy_prior"]}, outer="logo", inner_k=4, seed=1)
    raw = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, r.oof)
    assert raw == pytest.approx(0.0, abs=1e-12)                                            # perfectly anti-correlated, by construction
    m = P.noskill_metrics(y, r.oof, 0.5)
    assert m["roc_auc"] == pytest.approx(0.5) and m["loso_prior_artifact"]["roc_auc"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------------------------- the leakage trap
@pytest.mark.parametrize("seed", [100, 101, 102])
def test_row_level_split_is_fooled_by_subject_fingerprints_but_the_protocol_is_not(seed):
    """Labels are assigned to whole subjects at random, so NO model can honestly predict them from the features. A row-level K-fold
    still scores near-perfectly because it recognises the subject; the protocol must not."""
    df, _ = P.synthetic_disease(seed, n_subjects=20, rec=5, signal=0.0)
    X = df[D.DISEASE_FEATURES].reset_index(drop=True)
    y, g = df["status"].reset_index(drop=True), df["subject"].reset_index(drop=True)
    trap = ST.row_level_auc(X, y, seed)
    res = nested_group_cv(X, y, g, ST.lr_cands(), outer="logo", inner_k=4, seed=seed)
    honest = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, res.oof))
    assert trap > 0.95
    assert honest < 0.70                  # nowhere near the trap; pooled LOSO AUC on null data sits at or below 0.5 (Phase 3, S4)


@pytest.mark.parametrize("seed", [200, 201, 202])
def test_voice_noise_never_earns_the_registered_voice_claim_but_real_voice_signal_does(seed):
    noise = ST.risk_rep(seed, 0.0, 60, P.synthetic_risk)
    assert not (noise["lo"] is not None and noise["lo"] > 0)                               # no false 'voice helps' claim
    signal = ST.risk_rep(seed, 3.0, 60, P.synthetic_risk)
    assert signal["lo"] is not None and signal["lo"] > 0                                   # a real contribution is found
    assert signal["diff"] > noise["diff"] + 0.05


def test_voice_free_baseline_is_clearly_above_chance_when_the_label_is_built_from_elapsed_time(lr_only):
    """Guards the reason baselines exist. The label is slope*t >= 3 with a per-patient random slope, so time + baseline cannot be perfect,
    but it must be clearly above chance WITHOUT any voice column. 0.6 is a deliberately loose 'clearly above 0.5' bound (not tuned to the
    observed value)."""
    df, dcols, info = P.synthetic_risk(31, n_patients=16, visits=30, voice_signal=0.0)
    y, g = df["worsening"].reset_index(drop=True), df["subject#"].astype(str).reset_index(drop=True)
    res = nested_group_cv(df[["test_time", "baseline_total_UPDRS"]].reset_index(drop=True), y, g, ST.lr_cands(), outer=5, inner_k=3,
                          seed=0, group_level_target=False)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, res.oof) > 0.60


# ---------------------------------------------------------------------------------------------- E04: duplicate guard false positive
def test_demographic_only_baseline_does_not_crash_when_distinct_patients_share_age_and_sex(lr_only):
    """Regression (found by the protocol-integrity test): the exact-duplicate-row guard treated two different patients with the same
    (age, sex) as 'a recording duplicated across the boundary' and aborted the registered run at its first low-dimensional baseline."""
    df, dcols, info = P.synthetic_risk(41)
    assert df.groupby(["age", "sex"])["subject#"].nunique().max() > 1          # the fixture really has patients sharing (age, sex)
    out = P.run_risk(df, dcols, info, seeds=[1], n_boot=20, fast=True, evidence="SYNTHETIC_FIXTURE_NOT_EVIDENCE")
    assert "DEMOGRAPHIC_ONLY" in out["results_by_seed"][1]


def test_duplicate_guard_still_fires_for_the_voice_feature_sets():
    """The relaxation must not blind the guard where it matters: a recording duplicated across patients in the voice matrix still aborts."""
    df, dcols, info = P.synthetic_risk(42)
    a, b = df["subject#"].unique()[:2]
    donor = df[df["subject#"] == a].iloc[0].copy()
    df = df.copy()
    idx = df.index[df["subject#"] == b][0]
    df.loc[idx, :] = donor.to_numpy()
    df.loc[idx, "subject#"] = b                                                 # same feature vector, different patient
    from ml.evalkit import LeakageError
    with pytest.raises(LeakageError):
        P.run_risk(df, dcols, info, seeds=[1], n_boot=20, fast=True, evidence="E")
