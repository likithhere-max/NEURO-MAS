"""
Tests for ml/evalkit: leakage guards, nested CV, metrics with undefined handling, cluster bootstrap.

These use SYNTHETIC data purely to test code correctness (shapes, guards, reference values). Nothing here is
evidence about any clinical model's accuracy.
"""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evalkit import (LeakageError, aggregate_by_group, assert_group_disjoint, assert_group_labels_consistent,
                        assert_no_duplicate_rows, assert_temporal_order, cluster_bootstrap, evaluate_binary,
                        expected_calibration_error, group_kfold, group_label_permutation_test, guarded,
                        leave_one_group_out, nested_group_cv, ppv_npv_at_prevalence, regression_metrics,
                        score_metrics, temporal_group_holdout, threshold_metrics, wilson_interval,
                        youden_threshold)


def make_grouped(n_groups=24, rows=4, n_feat=5, signal=1.5, seed=0):
    rng = np.random.default_rng(seed)
    rows_ = []
    for gi in range(n_groups):
        label = gi % 2
        centre = rng.normal(0, 0.3, n_feat) + label * signal * np.r_[1, np.zeros(n_feat - 1)]
        for _ in range(rows):
            rows_.append({**{f"f{j}": centre[j] + rng.normal(0, 1.0) for j in range(n_feat)},
                          "g": f"S{gi:02d}", "y": label})
    return pd.DataFrame(rows_)


# ----------------------------------------------------------------------------- leakage guards
def test_random_kfold_on_grouped_data_is_rejected_by_the_guard():
    df = make_grouped()
    naive = KFold(5, shuffle=True, random_state=0).split(df)  # splits rows, ignoring patients
    with pytest.raises(LeakageError, match="group"):
        list(guarded(naive, groups=df["g"]))


def test_group_kfold_and_logo_never_share_a_group():
    df = make_grouped()
    for tr, te in group_kfold(df["g"], df["y"], n_splits=5, seed=1):
        assert not set(df["g"].iloc[tr]) & set(df["g"].iloc[te])
    n = 0
    for tr, te in leave_one_group_out(df["g"]):
        assert df["g"].iloc[te].nunique() == 1 and df["g"].iloc[te].iloc[0] not in set(df["g"].iloc[tr])
        n += 1
    assert n == df["g"].nunique()


def test_explicit_shared_group_raises():
    with pytest.raises(LeakageError):
        assert_group_disjoint(["a", "a", "b"], [0, 2], [1])


def test_duplicate_row_across_boundary_raises():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])  # row 2 duplicates row 0
    with pytest.raises(LeakageError, match="duplicate"):
        assert_no_duplicate_rows(X, [0, 1], [2])
    assert_no_duplicate_rows(X, [0, 1], [])  # nothing to leak


def test_future_to_past_leakage_is_rejected():
    t = [1, 2, 3, 4]
    assert_temporal_order(t, [0, 1], [2, 3])
    with pytest.raises(LeakageError, match="temporal"):
        assert_temporal_order(t, [0, 3], [1, 2])  # trains on t=4, tests on t=2,3


def test_temporal_group_holdout_is_forward_chaining_and_skips_short_histories():
    t = [1, 2, 3, 4, 5, 1, 2, 9]
    g = ["a", "a", "a", "a", "a", "b", "b", "c"]
    (tr, te), = list(temporal_group_holdout(t, g, test_fraction=0.4))
    assert set(np.array(g)[te]) <= {"a", "b"} and "c" not in set(np.array(g)[np.r_[tr, te]])  # c has 1 visit
    for grp in ("a", "b"):
        tt = np.array(t)
        assert tt[tr][np.array(g)[tr] == grp].max() < tt[te][np.array(g)[te] == grp].min()


def test_group_with_two_labels_is_rejected():
    with pytest.raises(LeakageError, match="more than one label"):
        assert_group_labels_consistent(["a", "a", "b"], [0, 1, 0])


def test_empty_split_is_rejected():
    with pytest.raises(LeakageError):
        list(guarded([(np.array([], int), np.array([0]))]))


# ----------------------------------------------------------------------------- nested CV does not touch held-out rows
FIT_LOG: list[set] = []


class SpyLR(BaseEstimator, ClassifierMixin):
    """Logistic regression that records the original row ids (feature 'rowid') it was fitted on."""

    def fit(self, X, y):
        FIT_LOG.append(set(X["rowid"].tolist()))
        self.cols_ = [c for c in X.columns if c != "rowid"]
        self.m_ = LogisticRegression(max_iter=500).fit(X[self.cols_], y)
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X[self.cols_])


def test_outer_test_rows_never_reach_any_fit_or_selection():
    FIT_LOG.clear()
    df = make_grouped(n_groups=12, rows=3)
    X = df.drop(columns=["g", "y"]).assign(rowid=np.arange(len(df)))
    cands = {"a": lambda: SpyLR(), "b": lambda: SpyLR()}
    res = nested_group_cv(X, df["y"], df["g"], cands, outer=4, inner_k=3, seed=3)
    # Reconstruct which rows were outer-test for each fold from the records
    test_rows_by_fold = [set(np.where(df["g"].isin(f.test_groups))[0].tolist()) for f in res.folds]
    assert len(FIT_LOG) > len(res.folds)  # inner fits + final fit per fold
    # every fit belongs to exactly one outer fold's training set: it must be disjoint from that fold's test rows
    for fit_rows in FIT_LOG:
        compatible = [t for t in test_rows_by_fold if not (fit_rows & t)]
        assert compatible, "a fit saw rows that are outer-test rows of EVERY fold"
    # stronger: the last fit of each fold (the refit on the full outer-train) excludes its own test rows
    per_fold = len(FIT_LOG) // len(res.folds)
    for k, t in enumerate(test_rows_by_fold):
        for fit_rows in FIT_LOG[k * per_fold:(k + 1) * per_fold]:
            assert not (fit_rows & t)


def test_feature_selection_outside_the_fold_inflates_auc_on_pure_noise_but_pipeline_selection_does_not():
    rng = np.random.default_rng(7)
    n_groups, rows, p = 40, 3, 400
    g = np.repeat(np.arange(n_groups), rows)
    y_group = rng.permutation(np.r_[np.zeros(n_groups // 2), np.ones(n_groups // 2)]).astype(int)
    y = y_group[g]
    X = pd.DataFrame(rng.normal(size=(len(g), p)), columns=[f"n{j}" for j in range(p)])  # NO signal at all

    # LEAKY: choose the 10 "best" features using ALL labels, then cross-validate.
    Xs = SelectKBest(f_classif, k=10).fit_transform(X, y)
    leaky = np.zeros(len(y))
    for tr, te in leave_one_group_out(g):
        leaky[te] = LogisticRegression(max_iter=500).fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
    from sklearn.metrics import roc_auc_score
    leaky_auc = roc_auc_score(y, leaky)

    # CORRECT: selection is a pipeline step, so it is re-fitted on each training split only.
    proper = np.zeros(len(y))
    for tr, te in leave_one_group_out(g):
        pipe = Pipeline([("s", StandardScaler()), ("k", SelectKBest(f_classif, k=10)),
                         ("c", LogisticRegression(max_iter=500))])
        proper[te] = pipe.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    proper_auc = roc_auc_score(y, proper)

    assert leaky_auc > 0.75, leaky_auc            # leakage manufactures "skill" from noise
    assert proper_auc < 0.65, proper_auc          # honest estimate is near chance
    assert leaky_auc - proper_auc > 0.2


def test_scaler_inside_pipeline_is_fitted_on_training_rows_only():
    seen = {}

    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None, **kw):
            seen["n"] = len(X)
            seen["max"] = float(np.max(X))
            return super().fit(X, y, **kw)

    X = np.vstack([np.zeros((9, 1)), [[1000.0]]])  # one extreme row, the last one
    y = np.array([0, 1] * 5)
    pipe = Pipeline([("s", RecordingScaler()), ("c", LogisticRegression())])
    pipe.fit(X[:9], y[:9])
    assert seen["n"] == 9 and seen["max"] == 0.0  # the 1000 never influenced the fitted scaler


def test_nested_cv_is_reproducible_with_fixed_seed_and_differs_in_structure_across_seeds():
    df = make_grouped(n_groups=16, rows=3, seed=5)
    X = df.drop(columns=["g", "y"])
    cands = {"lr": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=500))]),
             "lr_strong": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.01, max_iter=500))])}
    a = nested_group_cv(X, df["y"], df["g"], cands, outer=4, inner_k=3, seed=11)
    b = nested_group_cv(X, df["y"], df["g"], cands, outer=4, inner_k=3, seed=11)
    assert np.array_equal(a.oof, b.oof) and [f.chosen for f in a.folds] == [f.chosen for f in b.folds]
    assert np.all((a.oof >= 0) & (a.oof <= 1)) and not np.isnan(a.oof).any()


def test_nested_cv_rejects_non_dataframe_and_mixed_label_groups():
    df = make_grouped(n_groups=8, rows=2)
    with pytest.raises(TypeError):
        nested_group_cv(df.drop(columns=["g", "y"]).to_numpy(), df["y"], df["g"], {"a": lambda: LogisticRegression()})
    bad = df.copy()
    bad.loc[0, "y"] = 1 - bad.loc[0, "y"]
    with pytest.raises(LeakageError):
        nested_group_cv(bad.drop(columns=["g", "y"]), bad["y"], bad["g"], {"a": lambda: LogisticRegression()}, outer=3)


def test_youden_threshold_reference():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.9])
    # t=0.7: sens 2/3, spec 3/3 -> J=0.667 ; t=0.4: sens 1, spec 2/3 -> J=0.667 ; ties -> lowest threshold
    assert youden_threshold(y, p) == pytest.approx(0.4)
    assert youden_threshold(np.zeros(3), np.array([.1, .2, .3])) == 0.5  # undefined -> neutral default


# ----------------------------------------------------------------------------- metrics
def test_single_class_metrics_are_none_not_zero():
    m = score_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert m["roc_auc"] is None and m["pr_auc"] is None and "only one class" in m["undefined"]["roc_auc"]
    t = threshold_metrics([1, 1, 1], [1, 0, 1])
    assert t["specificity"] is None and t["balanced_accuracy"] is None and t["sensitivity"] == pytest.approx(2 / 3)
    assert "specificity" in t["undefined"]


def test_threshold_metrics_reference_values():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    t = threshold_metrics(y, pred)
    assert t["confusion"] == {"tp": 3, "fp": 1, "tn": 5, "fn": 1}
    assert t["sensitivity"] == pytest.approx(0.75) and t["specificity"] == pytest.approx(5 / 6)
    assert t["precision"] == pytest.approx(0.75) and t["f1"] == pytest.approx(0.75)
    assert t["balanced_accuracy"] == pytest.approx((0.75 + 5 / 6) / 2)


def test_wilson_interval_reference():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2


def test_ppv_npv_reference_and_domain():
    r = ppv_npv_at_prevalence(0.9, 0.8, 0.01)
    assert r["ppv"] == pytest.approx(0.009 / (0.009 + 0.198), rel=1e-9)  # = 0.04348
    assert r["npv"] == pytest.approx(0.792 / (0.792 + 0.001), rel=1e-9)
    # PPV is monotone increasing in prevalence
    ppvs = [ppv_npv_at_prevalence(0.9, 0.8, pr)["ppv"] for pr in (0.01, 0.05, 0.2, 0.75)]
    assert ppvs == sorted(ppvs)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ppv_npv_at_prevalence(0.9, 0.8, bad)
    with pytest.raises(ValueError):
        ppv_npv_at_prevalence(1.2, 0.8, 0.1)


def test_ece_and_brier_reference_values():
    y = np.array([1, 0] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(100, 0.8)) == pytest.approx(0.3)
    m = score_metrics(y, np.full(100, 0.5))
    assert m["brier"] == pytest.approx(0.25) and m["brier_constant_prevalence_baseline"] == pytest.approx(0.25)
    assert m["roc_auc"] == pytest.approx(0.5)


def test_probabilities_must_be_valid():
    for bad in ([0.1, float("nan")], [0.1, 1.2], [-0.1, 0.5]):
        with pytest.raises(ValueError):
            evaluate_binary([0, 1], bad)
    with pytest.raises(ValueError):
        evaluate_binary([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        evaluate_binary([0, 1], [0.1, 0.2], threshold=1.5)


def test_regression_metrics_reference():
    m = regression_metrics([1, 2, 3], [1, 2, 4])
    assert m["mae"] == pytest.approx(1 / 3) and m["rmse"] == pytest.approx(math.sqrt(1 / 3)) and m["r2"] == pytest.approx(0.5)
    c = regression_metrics([2, 2, 2], [2, 2, 3])
    assert c["r2"] is None and "zero variance" in c["undefined"]["r2"]
    with pytest.raises(ValueError):
        regression_metrics([1, np.nan], [1, 2])


def test_aggregate_by_group_requires_group_level_label():
    out = aggregate_by_group(["a", "a", "b"], [1, 1, 0], [0.2, 0.4, 0.9])
    assert out.set_index("group")["p"].to_dict() == {"a": pytest.approx(0.3), "b": 0.9}
    with pytest.raises(ValueError):
        aggregate_by_group(["a", "a"], [1, 0], [0.2, 0.4])


# ----------------------------------------------------------------------------- resampling
def test_cluster_bootstrap_is_wider_than_row_bootstrap_for_clustered_data():
    rng = np.random.default_rng(0)
    # 10 clusters; outcome constant within cluster (extreme intra-cluster correlation), 30 rows each
    df = pd.DataFrame({"c": np.repeat(np.arange(10), 30), "y": np.repeat(rng.integers(0, 2, 10), 30).astype(float)})
    stat = lambda d: float(d["y"].mean())  # noqa: E731
    cl = cluster_bootstrap(df, "c", stat, n_boot=400, seed=1)
    df["row"] = np.arange(len(df))
    rw = cluster_bootstrap(df, "row", stat, n_boot=400, seed=1)
    assert (cl["hi"] - cl["lo"]) > 2 * (rw["hi"] - rw["lo"])


def test_cluster_bootstrap_counts_degenerate_resamples_and_refuses_unreliable_intervals():
    df = pd.DataFrame({"c": [0, 1, 2], "y": [1, 1, 1], "p": [0.9, 0.8, 0.7]})
    out = cluster_bootstrap(df, "c", lambda d: None if d["y"].nunique() < 2 else 1.0, n_boot=100, seed=0)
    assert out["n_valid"] == 0 and out["lo"] is None and "degenerate" in out["note"]


def test_cluster_bootstrap_is_seed_reproducible():
    df = make_grouped(n_groups=10, rows=3)
    f = lambda d: float(d["f0"].mean())  # noqa: E731
    assert cluster_bootstrap(df, "g", f, n_boot=100, seed=4) == cluster_bootstrap(df, "g", f, n_boot=100, seed=4)


def test_group_label_permutation_rejects_row_level_targets_and_detects_signal():
    df = make_grouped(n_groups=20, rows=3, signal=3.0)
    X = df.drop(columns=["g", "y"])

    def p_fn(yp):
        oof = np.zeros(len(yp))
        for tr, te in leave_one_group_out(df["g"]):
            oof[te] = LogisticRegression(max_iter=500).fit(X.iloc[tr], yp[tr]).predict_proba(X.iloc[te])[:, 1]
        return oof

    from sklearn.metrics import roc_auc_score
    obs = roc_auc_score(df["y"], p_fn(df["y"].to_numpy()))
    r = group_label_permutation_test(df["g"], df["y"], p_fn, obs, n_perm=30, seed=0)
    assert obs > 0.9 and r["p_value"] < 0.1 and r["null_mean"] < 0.7
    bad_y = df["y"].to_numpy().copy()
    bad_y[0] = 1 - bad_y[0]
    with pytest.raises(ValueError):
        group_label_permutation_test(df["g"], bad_y, p_fn, 0.5, n_perm=2)
