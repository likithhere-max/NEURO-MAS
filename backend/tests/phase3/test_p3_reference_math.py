"""Phase 3 independent reference tests.

Every expected value below is derived BY HAND (or from a closed form / a different library / the Python standard
library) and written down as a literal or from an independently coded formula -- never by calling the function under
test to get its own expected answer.  The derivation is in the comment next to each case.
"""
import math
from statistics import NormalDist

import numpy as np
import pytest

from app.ml import longitudinal as L
from app.ml.personalized_anomaly import (
    detect_personal_anomaly, effective_z_threshold, t_critical, t_critical_95, _T_975,
)

N01 = NormalDist()


# ------------------------------------------------------------------ Student-t critical values
def test_t_table_matches_exact_quantile_to_table_precision():
    from scipy.stats import t
    for df, v in _T_975.items():
        assert abs(v - t.ppf(0.975, df)) < 5.0e-4, df          # table is quoted to 3 decimals


@pytest.mark.parametrize("df", [31, 40, 60, 120, 500, 5000])
def test_t_large_df_approximation_error_is_small(df):
    from scipy.stats import t
    assert abs(t_critical_95(df) - t.ppf(0.975, df)) < 2.0e-4


def test_t_critical_other_alpha_uses_exact_quantile():
    # df=2 has the closed form  t_{1-a/2} = sqrt(2 / (a * (2 - a)) ... ) : for a two-sided alpha the df=2 quantile is
    # t = (1-a) / sqrt(a * (2 - a) / 2)  [from F(t)=1/2 + t/(2 sqrt(2+t^2))].  alpha=0.1 -> 2.919986
    a = 0.10
    closed = (1 - a) / math.sqrt(a * (2 - a) / 2)
    assert t_critical(2, a) == pytest.approx(closed, rel=1e-9)


# ------------------------------------------------------------------ prediction interval, hand worked
def test_prediction_interval_hand_example():
    # history 10,12,11,13,9 : mean 11, sum sq dev = 1+1+0+4+4 = 10, s = sqrt(10/4) = 1.58113883
    # n=5 -> t_{.975,4}=2.776, PI half-width in SD units = 2.776*sqrt(1+1/5) = 2.776*1.0954451 = 3.04096
    h = [10, 12, 11, 13, 9]
    assert effective_z_threshold(5) == pytest.approx(2.776 * math.sqrt(1.2), rel=1e-12)
    hi = detect_personal_anomaly(h, 16)        # z = 5/1.58113883 = 3.16228 > 3.04096
    lo = detect_personal_anomaly(h, 14)        # z = 3/1.58113883 = 1.89737 < 3.04096
    assert hi["is_anomaly"] and hi["z_score"] == pytest.approx(3.1623, abs=1e-4)
    assert not lo["is_anomaly"] and lo["z_score"] == pytest.approx(1.8974, abs=1e-4)
    # exact edge: latest = mean + threshold*s  sits on the boundary -> flagged (rule is >=)
    s = math.sqrt(10 / 4)
    edge = 11 + effective_z_threshold(5) * s
    assert detect_personal_anomaly(h, edge + 1e-9)["is_anomaly"]
    assert not detect_personal_anomaly(h, edge - 1e-6)["is_anomaly"]


def test_prediction_interval_is_symmetric():
    h = [10, 12, 11, 13, 9]
    assert detect_personal_anomaly(h, 11 + 4.0)["is_anomaly"] == detect_personal_anomaly(h, 11 - 4.0)["is_anomaly"]


def test_insufficient_and_invalid_inputs_are_reported_not_guessed():
    assert detect_personal_anomaly([1, 2], 5)["method"] == "insufficient_history"
    assert detect_personal_anomaly([1, 2, 3], float("nan"))["method"] == "invalid_input"
    assert detect_personal_anomaly([1, 2, "x", None, float("inf"), 3, 4], 3)["readings_available"] == 4


# ------------------------------------------------------------------ Quesenberry Q (closed-form t CDFs)
def _t_cdf_df2(x):                       # F(x) = 1/2 + x / (2 sqrt(2 + x^2))
    return 0.5 + x / (2 * math.sqrt(2 + x * x))


def _t_cdf_df4(x):                       # F(x) = 1/2 + 3/8 * x/sqrt(1+x^2/4) * (1 - x^2/(12 (1+x^2/4)))
    u = 1 + x * x / 4
    return 0.5 + 0.375 * x / math.sqrt(u) * (1 - x * x / (12 * u))


def test_q_statistic_df1_exact_value_cauchy():
    # first defined Q is at index 2 (two prior points, df = m-1 = 1, the Cauchy law: F(x) = 1/2 + atan(x)/pi).
    # prior [1,2]: mean 1.5, s = 1/sqrt(2); x = 3 -> r = sqrt(2/3) * 1.5 * sqrt(2) = sqrt(3) ; atan(sqrt 3) = pi/3
    # F = 1/2 + 1/3 = 5/6  ->  Q = Phi^{-1}(5/6) = 0.967422
    q = L.q_statistics([1, 2, 3])
    assert q[:2] == [None, None] and q[2] == pytest.approx(N01.inv_cdf(5 / 6), abs=1e-9)


def test_q_statistic_df2_matches_closed_form():
    # prior = [1,2,3] (m=3): mean 2, s 1 ; new x = 10 -> r = sqrt(3/4)*(10-2)/1 ; df = m-1 = 2
    r = math.sqrt(3 / 4) * 8
    expected = N01.inv_cdf(_t_cdf_df2(r))
    q = L.q_statistics([1, 2, 3, 10])
    assert q[3] == pytest.approx(expected, abs=1e-9)


def test_q_statistic_df4_matches_closed_form():
    x = [10, 12, 11, 13, 9, 15]          # prior 5 points: mean 11, s = sqrt(2.5)
    r = math.sqrt(5 / 6) * (15 - 11) / math.sqrt(2.5)
    expected = N01.inv_cdf(_t_cdf_df4(r))
    assert L.q_statistics(x)[5] == pytest.approx(expected, abs=1e-9)


def test_q_statistic_is_antisymmetric_and_none_on_zero_variance():
    up = L.q_statistics([10, 12, 11, 13, 9, 15])[5]
    dn = L.q_statistics([10, 12, 11, 13, 9, 7])[5]          # 7 mirrors 15 around the prior mean 11
    assert dn == pytest.approx(-up, abs=1e-9)
    assert L.q_statistics([5, 5, 5, 5, 9])[4] is None       # prior has zero variance -> undefined, not +inf


def test_q_statistics_are_standard_normal_and_uncorrelated_under_iid_null():
    """The defining property (Quesenberry 1991): Q_i iid N(0,1) for iid Gaussian input."""
    from scipy.stats import kstest
    rng = np.random.default_rng(20261024)
    qs = []
    for _ in range(400):
        q = [v for v in L.q_statistics(rng.normal(50, 7, 30)) if v is not None]
        qs.append(q)
    flat = np.concatenate(qs)
    assert kstest(flat, "norm").pvalue > 0.001                       # distribution
    assert abs(flat.mean()) < 0.03 and abs(flat.std(ddof=1) - 1) < 0.03
    a = np.concatenate([q[:-1] for q in qs]); b = np.concatenate([q[1:] for q in qs])
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.04                       # serial independence


# ------------------------------------------------------------------ CUSUM / EWMA hand cases
def test_cusum_boundary_is_strict_and_hand_timed():
    # constant z=1, k=.5 -> C+ = .5, 1.0, ..., 5.0 at index 9 (NOT > 5), 5.5 at index 10 -> first alarm index 10
    r = L.cusum([1.0] * 12, k=0.5, h=5.0)
    assert r["C_plus"][9] == 5.0 and r["first_alarm_index"] == 10 and r["alarm_direction"] == "up"
    assert L.cusum([-1.0] * 12, 0.5, 5.0)["alarm_direction"] == "down"


def test_cusum_resets_at_zero_hand_values():
    # z = 1,1,1,1 -> C+ = 2.0 ; then z=-3 : C+ = max(0, 2-3-.5)=0 ; C- = max(0, 0+3-.5)=2.5
    r = L.cusum([1, 1, 1, 1, -3], 0.5, 5.0)
    assert r["C_plus"][3] == 2.0 and r["C_plus"][4] == 0.0 and r["C_minus"][4] == 2.5


def test_ewma_limits_hand_values():
    # lim_t = L * sqrt(lam/(2-lam) * (1-(1-lam)^(2t))) ; lam=.2, L=3 : t=1 -> 3*sqrt(.2/1.8*.36)=0.6
    r = L.ewma([0.0] * 3, 0.2, 3.0)
    assert r["limits"][0] == pytest.approx(0.6, abs=1e-12)
    assert r["limits"][1] == pytest.approx(3 * math.sqrt(0.2 / 1.8 * (1 - 0.64 ** 2)), abs=1e-12)
    # asymptote  L*sqrt(lam/(2-lam)) = 3*sqrt(1/9) = 1.0
    assert L.ewma([0.0] * 200, 0.2, 3.0)["final_limit"] == pytest.approx(1.0, abs=1e-9)


def test_ewma_first_point_alarm_threshold_is_z_eq_3():
    # w1 = lam*z = .2 z must exceed .6  <=>  z > 3
    assert L.ewma([3.01], 0.2, 3.0)["first_alarm_index"] == 0
    assert L.ewma([2.99], 0.2, 3.0)["first_alarm_index"] is None


def test_ewma_cannot_alarm_on_a_sustained_shift_of_exactly_L_sqrt_lambda_over_2_minus_lambda():
    # w_t -> delta, limit_t -> L*sqrt(lam/(2-lam)) = 1.0 : a persistent +1.0-SD shift of the monitored statistic
    # never exceeds the limit (w_t = 1-.8^t < sqrt(1-.64^t) for all t), whereas +1.5 does.
    assert L.ewma([1.0] * 500, 0.2, 3.0)["first_alarm_index"] is None
    assert L.ewma([1.5] * 60, 0.2, 3.0)["first_alarm_index"] is not None


# ------------------------------------------------------------------ Mann-Kendall
def test_mann_kendall_hand_values_monotone():
    # y=1..5: S=10, var=n(n-1)(2n+5)/18 = 5*4*15/18 = 16.6667, z=(10-1)/4.0825=2.2045, p=2(1-Phi(z))
    r = L.mann_kendall(np.array([1, 2, 3, 4, 5.0]), autocorr_correct=False)
    z = 9 / math.sqrt(300 / 18)
    assert r["S"] == 10 and r["var_S"] == pytest.approx(300 / 18) and r["z"] == pytest.approx(z)
    assert r["p_value"] == pytest.approx(2 * (1 - N01.cdf(z)), abs=1e-9)


def test_mann_kendall_hand_values_with_ties():
    # y=[1,1,2,3]: pairs (1,1)=0,(1,2)+,(1,3)+,(1,2)+,(1,3)+,(2,3)+ -> S=5 ; tie term 2*1*(2*2+5)=18
    # var=(4*3*13-18)/18=7.6667 ; z=(5-1)/sqrt(7.6667)=1.4446 ; p=0.1486
    r = L.mann_kendall(np.array([1, 1, 2, 3.0]), autocorr_correct=False)
    assert r["S"] == 5 and r["var_S"] == pytest.approx(138 / 18)
    assert r["z"] == pytest.approx(4 / math.sqrt(138 / 18))
    assert r["p_value"] == pytest.approx(2 * (1 - N01.cdf(4 / math.sqrt(138 / 18))), abs=1e-9)


def test_mann_kendall_S_agrees_with_scipy_kendall_tau():
    from scipy.stats import kendalltau
    rng = np.random.default_rng(3)
    y = rng.normal(size=25) + np.arange(25) * 0.1
    n = len(y)
    tau = kendalltau(np.arange(n), y).statistic            # no ties -> tau_b == tau_a
    assert L.mann_kendall(y, autocorr_correct=False)["S"] == pytest.approx(tau * n * (n - 1) / 2)


def test_mann_kendall_sign_symmetry_and_constant_series():
    y = np.array([3, 1, 4, 1, 5, 9, 2, 6.0])
    a, b = L.mann_kendall(y, autocorr_correct=False), L.mann_kendall(-y, autocorr_correct=False)
    assert a["S"] == -b["S"] and a["p_value"] == pytest.approx(b["p_value"])
    assert L.mann_kendall(np.ones(6))["status"] == "not_assessed"


def test_mann_kendall_autocorrelation_inflation_hand_value():
    # residuals 1,1,1,-1,-1,-1 : sum r_t r_{t+1} = 1+1-1+1+1 = 3 ; sum r^2 = 6 ; rho = .5 ; inflation (1+.5)/(1-.5)=3
    y = np.arange(6.0)
    r = L.mann_kendall(y, residuals=np.array([1, 1, 1, -1, -1, -1.0]))
    assert r["lag1_autocorr_residuals"] == pytest.approx(0.5) and r["variance_inflation"] == pytest.approx(3.0)
    # negative rho is NOT used to deflate the variance (conservative)
    r2 = L.mann_kendall(y, residuals=np.array([1, -1, 1, -1, 1, -1.0]))
    assert r2["variance_inflation"] == 1.0


def test_theil_sen_matches_brute_force_median_of_pairwise_slopes():
    t = np.array([0, 1, 2, 4, 7, 8.0]); y = np.array([0.1, 2.3, 3.9, 8.5, 13.7, 17.2])
    slopes = [(y[j] - y[i]) / (t[j] - t[i]) for i in range(len(t)) for j in range(i + 1, len(t))]
    assert L.theil_sen(t, y)["slope_per_day"] == pytest.approx(float(np.median(slopes)))
    # robustness: one gross outlier moves OLS but barely moves the Theil-Sen slope
    y2 = y.copy(); y2[2] = 100.0
    assert abs(L.theil_sen(t, y2)["slope_per_day"] - L.theil_sen(t, y)["slope_per_day"]) < 0.5


# ------------------------------------------------------------------ robust scale
def test_mad_constant_is_the_gaussian_consistency_factor():
    assert L.MAD_TO_SIGMA == pytest.approx(1 / N01.inv_cdf(0.75), rel=1e-12)
    med, sig, how = L.robust_location_scale(np.array([1, 2, 3, 4, 100.0]))   # median 3, |dev| = 2,1,0,1,97 -> MAD 1
    assert med == 3.0 and sig == pytest.approx(1.4826022185056018) and how == "median/MAD"
    assert L.robust_location_scale(np.array([2.0, 2, 2, 2]))[2] == "zero_variance"


# ------------------------------------------------------------------ sampling / cleaning
def test_clean_series_sorts_merges_and_drops():
    t, y, q = L.clean_series([3, 1, 2, 2, 5], [30, 10, 20, 22, float("nan")])
    assert t.tolist() == [1.0, 2.0, 3.0] and y.tolist() == [10.0, 21.0, 30.0]          # median(20,22)=21
    assert q == {"dropped_invalid": 1, "merged_duplicate_timestamps": 1}
    with pytest.raises(ValueError):
        L.clean_series([1, 2], [1])


def test_boolean_and_overflow_values_are_invalid_not_numbers():
    _, y, q = L.clean_series([1, 2, 3], [True, 1e12, 5.0])
    assert y.tolist() == [5.0] and q["dropped_invalid"] == 2
