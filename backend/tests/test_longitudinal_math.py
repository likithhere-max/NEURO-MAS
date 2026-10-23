"""
Independent-reference tests for app/ml/longitudinal.py (Phase 2, C2/C3/C5).
Expected values are computed BY HAND (shown in comments) or taken from scipy as an independent implementation.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import stats

from app.ml import longitudinal as L


# ----------------------------------------------------------------------------- Theil-Sen
def test_theil_sen_hand_example():
    # points (0,0),(1,1),(2,5): pairwise slopes (0,1)=1, (0,2)=2.5, (1,2)=4 -> median 2.5
    r = L.theil_sen(np.array([0., 1, 2]), np.array([0., 1, 5]))
    assert r["slope_per_day"] == pytest.approx(2.5)


def test_theil_sen_ignores_a_gross_outlier_that_ruins_least_squares():
    t = np.arange(10.0)
    y = 3.0 * t + 1.0
    y[9] = 500.0
    assert L.theil_sen(t, y)["slope_per_day"] == pytest.approx(3.0)
    assert np.polyfit(t, y, 1)[0] > 20  # OLS is wrecked: shows why a robust estimator is used


def test_theil_sen_irregular_times_and_matches_scipy():
    t = np.array([0.0, 0.5, 3.0, 3.2, 9.0, 15.0])
    y = np.array([1.0, 1.2, 2.4, 2.5, 5.1, 7.9])
    ref = stats.theilslopes(y, t)[0]
    assert L.theil_sen(t, y)["slope_per_day"] == pytest.approx(ref)


def test_theil_sen_degenerate_inputs():
    assert L.theil_sen(np.array([1., 1, 1]), np.array([1., 2, 3]))["status"] == "not_assessed"  # zero time span
    assert L.theil_sen(np.array([1., 2]), np.array([1., 2]))["status"] == "not_assessed"


# ----------------------------------------------------------------------------- Mann-Kendall
def test_mann_kendall_hand_example_monotone_five():
    # S = 10 ; Var = 5*4*15/18 = 16.667 ; Z = (10-1)/sqrt(16.667) = 2.2045 ; p = erfc(Z/sqrt2) = 0.02748
    r = L.mann_kendall(np.array([1., 2, 3, 4, 5]), autocorr_correct=False)
    assert r["S"] == 10 and r["var_S"] == pytest.approx(50 / 3)
    assert r["z"] == pytest.approx(9 / math.sqrt(50 / 3)) and r["p_value"] == pytest.approx(0.02748, abs=2e-5)


def test_mann_kendall_tie_correction_hand_example():
    # [1,1,2,2]: S=4 ; Var = (4*3*13 - 2*(2*1*9))/18 = (156-36)/18 = 6.667 ; Z = 3/sqrt(6.667) = 1.1619
    r = L.mann_kendall(np.array([1., 1, 2, 2]), autocorr_correct=False)
    assert r["S"] == 4 and r["var_S"] == pytest.approx(120 / 18) and r["z"] == pytest.approx(3 / math.sqrt(120 / 18))


def test_mann_kendall_sign_flips_with_direction_and_constant_is_unassessed():
    y = np.array([3., 1, 4, 1, 5, 9, 2, 6])
    up, down = L.mann_kendall(y, autocorr_correct=False), L.mann_kendall(-y, autocorr_correct=False)
    assert up["S"] == -down["S"] and up["z"] == pytest.approx(-down["z"]) and up["p_value"] == pytest.approx(down["p_value"])
    assert L.mann_kendall(np.full(6, 2.0))["status"] == "not_assessed"


def test_mann_kendall_p_matches_kendall_tau_for_untied_data_asymptotically():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(0.1, 1, 60))
    mk = L.mann_kendall(y, autocorr_correct=False)
    tau, p_scipy = stats.kendalltau(np.arange(60), y, method="asymptotic")
    assert mk["p_value"] == pytest.approx(p_scipy, rel=0.03)  # same normal approximation up to the continuity detail


def test_autocorrelation_correction_only_widens_for_positive_rho():
    y = np.arange(12.0) + np.random.default_rng(2).normal(0, 0.3, 12)
    resid_pos = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1.0])      # strongly positive lag-1
    resid_neg = np.array([1, -1] * 6, float)                                # negative lag-1
    base = L.mann_kendall(y, autocorr_correct=False)
    pos, neg = L.mann_kendall(y, residuals=resid_pos), L.mann_kendall(y, residuals=resid_neg)
    rho = (resid_pos[:-1] * resid_pos[1:]).sum() / (resid_pos ** 2).sum()
    assert pos["variance_inflation"] == pytest.approx((1 + rho) / (1 - rho)) and pos["p_value"] > base["p_value"]
    assert neg["variance_inflation"] == 1.0 and neg["p_value"] == pytest.approx(base["p_value"])


# ----------------------------------------------------------------------------- CUSUM / EWMA
def test_cusum_hand_example():
    # k=.5: z=[.5,1.5,2,0,-3] -> C+ = [0,1,2.5,2,0] ; C- = [0,0,0,0,2.5]
    r = L.cusum([0.5, 1.5, 2.0, 0.0, -3.0], k=0.5, h=100)
    assert r["C_plus"] == pytest.approx([0, 1.0, 2.5, 2.0, 0.0]) and r["C_minus"] == pytest.approx([0, 0, 0, 0, 2.5])
    assert r["first_alarm_index"] is None
    r2 = L.cusum([0.5, 1.5, 2.0, 0.0, -3.0], k=0.5, h=2.0)
    assert r2["first_alarm_index"] == 2 and r2["alarm_direction"] == "up"


def test_cusum_is_zero_for_values_inside_the_reference_band():
    r = L.cusum([0.4, -0.4, 0.1, -0.2] * 5, k=0.5, h=1.0)
    assert max(r["C_plus"]) == 0 and max(r["C_minus"]) == 0


def test_ewma_hand_example_and_limits():
    # lam=.2, x=1,1,1 : 0.2, 0.36, 0.488 ; limit_t = 3*sqrt(.2/1.8*(1-.64^t))  -> t=1 : 3*sqrt(.04)=0.6
    r = L.ewma([1, 1, 1], lam=0.2, L=3.0)
    assert r["path"] == pytest.approx([0.2, 0.36, 0.488]) and r["limits"][0] == pytest.approx(0.6)
    assert r["limits"][1] == pytest.approx(3 * math.sqrt(0.2 / 1.8 * (1 - 0.8 ** 4)))
    # asymptotic limit L*sqrt(lam/(2-lam))
    far = L.ewma([0] * 200, 0.2, 3.0)["limits"][-1]
    assert far == pytest.approx(3 * math.sqrt(0.2 / 1.8), rel=1e-6)
    assert L.ewma([5, 5, 5, 5], 0.5, 3.0)["first_alarm_index"] is not None


# ----------------------------------------------------------------------------- Q statistics (self-starting)
def test_q_statistics_hand_example():
    # x=[10,12,14] : prior [10,12] mean 11, s=sqrt(2)=1.4142, m=2 -> r = sqrt(2/3)*(14-11)/1.4142 = 1.7321
    # Q = Phi^-1( F_t1(1.7321) ) ; F_t1(x)=0.5+atan(x)/pi => 0.5+atan(1.7321)/pi = 0.5+ (pi/3)/pi = 0.8333
    q = L.q_statistics([10.0, 12.0, 14.0])
    assert q[:2] == [None, None]
    assert q[2] == pytest.approx(stats.norm.ppf(5 / 6), abs=1e-9)


def test_q_statistics_are_standard_normal_and_uncorrelated_under_the_null():
    rng = np.random.default_rng(11)
    Q = np.array([L.q_statistics(rng.normal(5, 2, 12))[2:] for _ in range(6000)])  # columns i=3..12
    assert abs(Q.mean()) < 0.03 and Q.std() == pytest.approx(1.0, abs=0.03)
    assert stats.kstest(Q[:, 7], "norm").pvalue > 0.001
    c = np.corrcoef(Q[:, 3], Q[:, 8])[0, 1]
    assert abs(c) < 0.05  # independence, the property that makes the chart's error rate exact


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=5, max_size=15, unique=True),
       st.floats(0.5, 20), st.floats(-100, 100))
def test_property_q_is_location_scale_invariant_and_odd(xs, a, b):
    base = L.q_statistics(xs)
    mapped = L.q_statistics([a * v + b for v in xs])
    neg = L.q_statistics([-v for v in xs])
    for u, v, w in zip(base, mapped, neg):
        if u is None:
            continue
        assert v is not None and v == pytest.approx(u, abs=1e-6)
        assert w == pytest.approx(-u, abs=1e-6)


def test_q_statistics_zero_variance_history_is_undefined_not_infinite():
    q = L.q_statistics([5.0, 5.0, 5.0, 9.0])
    assert q[2] is None and q[3] is None


# ----------------------------------------------------------------------------- series cleaning
def test_clean_series_sorts_parses_iso_and_merges_duplicates():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [(t0 + timedelta(days=2)).isoformat().replace("+00:00", "Z"), t0, t0 + timedelta(seconds=10), "garbage", t0 + timedelta(days=1)]
    vals = [3.0, 1.0, 1.4, 9.0, float("nan")]
    t, y, q = L.clean_series(times, vals)
    assert len(t) == 2 and np.all(np.diff(t) > 0)
    assert y.tolist() == [pytest.approx(1.2), 3.0]            # median of the two same-minute readings 1.0 and 1.4
    assert q == {"dropped_invalid": 2, "merged_duplicate_timestamps": 1}


def test_clean_series_rejects_length_mismatch_and_bool_values():
    with pytest.raises(ValueError):
        L.clean_series([1, 2], [1])
    _, y, q = L.clean_series([1, 2, 3], [True, 2.0, 3.0])
    assert len(y) == 2 and q["dropped_invalid"] == 1


# ----------------------------------------------------------------------------- end to end analysis
def noisy(n, seed, sd=1.0, mean=10.0):
    return list(mean + np.random.default_rng(seed).normal(0, sd, n))


def test_insufficient_and_invalid_inputs_are_labelled_not_guessed():
    r = L.analyze_series("tremor_amplitude", [0, 1], [1, 2])
    assert r.detection_status == "insufficient_data" and r.signals == [] and r.is_diagnosis is False
    r = L.analyze_series("tremor_amplitude", [0, 1, 2], [1, 2, 3, 4])  # length mismatch
    assert r.detection_status == "invalid_input"


def test_report_schema_is_complete_and_never_a_diagnosis():
    r = L.analyze_series("tremor_amplitude", list(range(30)), noisy(30, 5)).as_dict()
    for key in ("method_version", "baseline", "observation_count", "sampling", "statistics", "thresholds", "signals",
                "data_quality_warnings", "clinical_significance", "uncertainty_and_limitations", "detection_status", "is_diagnosis"):
        assert key in r
    assert r["is_diagnosis"] is False and r["clinical_significance"]["assessed"] is False
    assert "no clinician-approved" in r["clinical_significance"]["reason"]
    assert r["baseline"]["definition"].startswith("first 5")
    assert r["thresholds"]["point_t_interval"]["provenance"]


def find_seed(pred, limit=200):
    for s in range(limit):
        if pred(s):
            return s
    raise AssertionError("no seed found")


def test_level_shift_is_detected_and_classified_by_metric_polarity():
    def build(seed):
        y = noisy(36, seed)
        for i in range(24, 36):
            y[i] += 5.0
        return y
    seed = find_seed(lambda s: any(x["method"] in ("cusum", "ewma") for x in L.analyze_series("tremor_amplitude", range(36), build(s)).signals))
    y = build(seed)
    worse = L.analyze_series("tremor_amplitude", list(range(36)), y)         # higher_worse: up = adverse
    better = L.analyze_series("gait_speed_m_s", list(range(36)), y)          # lower_worse: up = favourable
    unknown = L.analyze_series("mystery_metric", list(range(36)), y)
    assert worse.detection_status == "signal_adverse"
    assert better.detection_status == "signal_favourable"
    assert unknown.detection_status == "signal_direction_unknown"
    assert any(s["kind"] == "level_shift" and s["direction"] == "up" for s in worse.signals)


def test_downward_shift_in_a_lower_is_worse_metric_is_adverse():
    def build(seed):
        y = noisy(36, seed, mean=1.2, sd=0.05)
        for i in range(24, 36):
            y[i] -= 0.35
        return y
    seed = find_seed(lambda s: L.analyze_series("gait_speed_m_s", range(36), build(s)).detection_status == "signal_adverse")
    assert L.analyze_series("gait_speed_m_s", range(36), build(seed)).detection_status == "signal_adverse"


def test_clear_linear_trend_triggers_trend_signal_with_slope_and_ci():
    t = list(range(30))
    y = [10 + 0.3 * i + e for i, e in zip(t, np.random.default_rng(3).normal(0, 0.8, 30))]
    r = L.analyze_series("tremor_amplitude", t, y)
    trend = [s for s in r.signals if s["kind"] == "trend"]
    assert trend and trend[0]["direction"] == "up" and trend[0]["adverse"] is True
    ts = r.statistics["theil_sen"]
    assert ts["slope_ci_lo"] < 0.3 < ts["slope_ci_hi"] and ts["slope_per_day"] == pytest.approx(0.3, abs=0.1)


def test_flat_noise_series_has_no_trend_signal_and_reports_short_series_warning_when_short():
    r = L.analyze_series("tremor_amplitude", list(range(7)), noisy(7, 9))
    assert any("short_series" in w for w in r.data_quality_warnings)
    assert r.statistics["theil_sen"]["status"] == "not_assessed"
    assert r.statistics["cusum"]["status"] == "not_assessed"


def test_irregular_sampling_is_flagged():
    t = [0, 1, 2, 40, 41, 100, 101, 102, 190, 191]
    r = L.analyze_series("tremor_amplitude", t, noisy(10, 4))
    assert r.sampling["irregular"] is True and any("irregular_sampling" in w for w in r.data_quality_warnings)


def test_mcid_is_only_used_when_configured():
    y = noisy(12, 6)
    base = L.analyze_series("tremor_amplitude", range(12), y)
    cfg = L.AnalysisConfig(mcid=0.5, mcid_provenance="TEST FIXTURE ONLY - not a clinical value")
    with_mcid = L.analyze_series("tremor_amplitude", range(12), y, cfg)
    assert base.clinical_significance["assessed"] is False
    assert with_mcid.clinical_significance["assessed"] is True and "observed_change_vs_mcid" in with_mcid.clinical_significance


def test_deterministic_output():
    y = noisy(25, 8)
    assert L.analyze_series("tremor_amplitude", range(25), y).as_dict() == L.analyze_series("tremor_amplitude", range(25), y).as_dict()


def test_zero_variance_series_does_not_crash():
    r = L.analyze_series("tremor_amplitude", range(10), [5.0] * 10)
    assert r.detection_status in ("no_signal", "signal_adverse", "signal_favourable", "signal_direction_unknown")
    assert any("baseline_zero_variance" in w for w in r.data_quality_warnings)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.integers(-10**6, 10**6)), max_size=40))
def test_property_never_raises_and_status_is_always_valid(vals):
    r = L.analyze_series("tremor_amplitude", list(range(len(vals))), vals)
    assert r.detection_status in {"insufficient_data", "invalid_input", "no_signal", "signal_adverse", "signal_favourable", "signal_direction_unknown"}
    assert r.is_diagnosis is False


@settings(max_examples=25, deadline=None)
@given(st.lists(st.integers(0, 100), min_size=12, max_size=24, unique=True), st.floats(0.5, 10), st.floats(-50, 50))
def test_property_signals_are_invariant_to_affine_rescaling(vals, a, b):
    assume(len(set(vals)) > 8)
    t = list(range(len(vals)))
    r1 = L.analyze_series("mystery_metric", t, [float(v) for v in vals])
    r2 = L.analyze_series("mystery_metric", t, [a * v + b for v in vals])
    key = lambda r: sorted((s["method"], s["kind"], s["direction"]) for s in r.signals)  # noqa: E731
    assert key(r1) == key(r2)
