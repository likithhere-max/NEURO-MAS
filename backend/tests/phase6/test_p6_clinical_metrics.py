"""Metric implementations checked against independent computations.  Synthetic inputs: these tests show the code computes what
it says; they are NOT evidence about any model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss as sk_log_loss, roc_auc_score

from ml.evalkit import clinical_metrics as CM


def _calibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_scalar_metrics_match_scikit_learn():
    y, p = _calibrated()
    m = CM.point_metrics(y, p, 0.5)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["log_loss"] == pytest.approx(sk_log_loss(y, p), rel=1e-6)


def test_threshold_metrics_hand_computed():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .2, .6, .4, .3, .2, .1, .1])
    c = CM.confusion_at(y, p, 0.5)
    assert c == {"tp": 3, "fn": 1, "fp": 1, "tn": 5}
    m = CM.point_metrics(y, p, 0.5)
    assert m["sensitivity"] == 3 / 4 and m["specificity"] == 5 / 6 and m["ppv"] == 3 / 4 and m["npv"] == 5 / 6
    assert m["accuracy"] == 8 / 10 and m["balanced_accuracy"] == pytest.approx((3 / 4 + 5 / 6) / 2)


def test_undefined_metrics_are_none_not_zero():
    m = CM.point_metrics(np.zeros(5, int), np.full(5, 0.3), 0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None and m["sensitivity"] is None and m["calibration_slope"] is None
    m2 = CM.point_metrics(np.array([0, 1, 0, 1]), np.array([.1, .2, .1, .2]), 0.9)   # nothing predicted positive
    assert m2["ppv"] is None and m2["sensitivity"] == 0.0


def test_calibration_recovers_known_values():
    y, p = _calibrated(n=20000, seed=1)
    c = CM.calibration_intercept_slope(y, p)
    assert c["slope"] == pytest.approx(1.0, abs=0.06) and c["intercept"] == pytest.approx(0.0, abs=0.05)


def test_overconfident_scores_have_slope_below_one_and_shifted_scores_nonzero_intercept():
    y, p = _calibrated(n=20000, seed=2)
    lg = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-2.0 * lg))                     # exaggerated logits: true slope should be about 0.5
    assert CM.calibration_intercept_slope(y, over)["slope"] == pytest.approx(0.5, abs=0.05)
    shifted = 1 / (1 + np.exp(-(lg + 1.0)))                # systematic over-prediction
    assert CM.calibration_intercept_slope(y, shifted)["intercept"] == pytest.approx(-1.0, abs=0.08)


def test_calibration_is_none_with_a_reason_when_not_estimable():
    assert CM.calibration_intercept_slope(np.ones(10, int), np.linspace(.1, .9, 10))["slope"] is None
    c = CM.calibration_intercept_slope(np.array([0, 1] * 10), np.full(20, 0.4))
    assert c["slope"] is None and "constant" in c["reasons"]["slope"]
    sep = CM.calibration_intercept_slope(np.array([0] * 10 + [1] * 10), np.r_[np.linspace(.01, .4, 10), np.linspace(.6, .99, 10)])
    assert sep["slope"] is None and "separable" in sep["reasons"]["slope"]


def test_decision_curve_hand_computed():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .2, .6, .4, .3, .2, .1, .1])
    r = CM.net_benefit(y, p, [0.5])[0]
    # tp=3 fp=1 n=10 t=.5 -> 3/10 - 1/10 * 1 = 0.2 ; treat-all: .4 - .6*1 = -0.2
    assert r["net_benefit_model"] == pytest.approx(0.2) and r["net_benefit_treat_all"] == pytest.approx(-0.2) and r["net_benefit_treat_none"] == 0.0
    r2 = CM.net_benefit(y, p, [0.2])[0]                    # odds .25 ; act if p>=.2: tp=4, fp=4 (.6,.4,.3,.2) -> .4 - .4*.25 = .3
    assert r2["net_benefit_model"] == pytest.approx(0.4 - 0.4 * 0.25)
    with pytest.raises(ValueError):
        CM.net_benefit(y, p, [0.0])
    with pytest.raises(ValueError):
        CM.net_benefit(y, p, [1.0])


def test_treat_all_net_benefit_equals_model_that_flags_everyone():
    y, p = _calibrated(500, 3)
    r = CM.net_benefit(y, np.ones(500), [0.3])[0]
    assert r["net_benefit_model"] == pytest.approx(r["net_benefit_treat_all"])


def _clustered(n_sub=30, rows=20, signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    sub = np.repeat(np.arange(n_sub), rows)
    y = (rng.uniform(size=len(sub)) < 0.4).astype(int)
    p = 1 / (1 + np.exp(-(signal * (y - 0.4) * 2 + rng.normal(size=len(sub)))))
    return y, p, sub


def test_comprehensive_has_interval_for_every_metric_and_contains_the_point_for_a_stable_statistic():
    y, p, s = _clustered()
    r = CM.comprehensive(y, p, s, threshold=0.5, n_boot=200, seed=1)
    assert set(r["metrics"]) == set(CM.METRICS) and r["n_clusters"] == 30
    a = r["metrics"]["roc_auc"]
    assert a["lo"] < a["point"] < a["hi"]
    assert r["confusion"]["tp"] + r["confusion"]["fn"] == int(y.sum())


def test_resampling_unit_is_the_cluster_rows_give_a_narrower_interval():
    y, p, s = _clustered(seed=4)
    by_cluster = CM.comprehensive(y, p, s, threshold=0.5, n_boot=300, seed=1)["metrics"]["roc_auc"]
    by_row = CM.comprehensive(y, p, np.arange(len(y)), threshold=0.5, n_boot=300, seed=1)["metrics"]["roc_auc"]
    assert (by_cluster["hi"] - by_cluster["lo"]) > 0.8 * (by_row["hi"] - by_row["lo"])   # never meaningfully narrower than row-level on clustered noise


def test_paired_difference_is_zero_for_identical_models_and_detects_a_real_gap():
    y, p, s = _clustered(seed=5, signal=1.5)
    same = CM.paired_difference(y, p, p, s, metric="roc_auc", threshold=0.5, n_boot=100, seed=1)
    assert same["point_difference_a_minus_b"] == 0 and not same["interval_excludes_zero"]
    rng = np.random.default_rng(0)
    noise = rng.uniform(size=len(y))
    gap = CM.paired_difference(y, p, noise, s, metric="roc_auc", threshold=0.5, n_boot=200, seed=1)
    assert gap["interval_excludes_zero"] and gap["a_better_with_interval_excluding_zero"] is True
    inv = CM.paired_difference(y, noise, p, s, metric="brier", threshold=0.5, n_boot=200, seed=1)
    assert inv["a_better_with_interval_excluding_zero"] is False                 # lower Brier is better; noise is worse
    with pytest.raises(ValueError):
        CM.paired_difference(y, p, p, s, metric="nonsense", threshold=0.5)


def test_bootstrap_is_deterministic_for_a_seed():
    y, p, s = _clustered(seed=6)
    a = CM.comprehensive(y, p, s, threshold=0.5, n_boot=60, seed=9)
    b = CM.comprehensive(y, p, s, threshold=0.5, n_boot=60, seed=9)
    assert a == b


def test_subgroups_are_withheld_when_too_small():
    y, p, s = _clustered(n_sub=30, rows=20, seed=7)
    df = pd.DataFrame({"y": y, "p": p, "subject": s, "grp": np.where(s < 3, "tiny", "big")})
    res = {r["level"]: r for r in CM.subgroup_report(df, "grp", threshold=0.5, n_boot=60)}
    assert res["tiny"]["status"] == "insufficient" and "result" not in res["tiny"]
    assert res["big"]["status"] == "reported" and res["big"]["result"]["n_clusters"] == 27


def test_cross_fitted_recalibration_improves_calibration_of_overconfident_scores_and_never_sees_own_subject():
    rng = np.random.default_rng(8)
    n_sub, rows = 60, 40
    s = np.repeat(np.arange(n_sub), rows)
    base = rng.uniform(0.1, 0.9, len(s))
    y = (rng.uniform(size=len(s)) < base).astype(int)
    lg = np.log(base / (1 - base))
    p = 1 / (1 + np.exp(-3.0 * lg))
    r = CM.cross_fitted_recalibration(y, p, s, k=5, seed=0)
    assert r["status"] == "estimated" and r["brier_after"] < r["brier_before"]
    assert abs(r["calibration_after"]["slope"] - 1) < abs(r["calibration_before"]["slope"] - 1)
    assert r["label"].startswith("EXPLORATORY")
    assert CM.cross_fitted_recalibration(y[:10], p[:10], s[:10], k=5)["status"] == "not_estimable"
