"""
Personalized within-patient longitudinal anomaly detection (Roadmap item 6,
Track 1). Pure statistics on ONE patient's own historical readings for a
single metric -- deliberately NOT a trained model and NOT a population
classifier.

Audit fix (High): the original rule flagged |z| >= 2 using the sample SD of
the history. For a NEW observation that is not a valid 95% test: it ignores
(a) the uncertainty of the estimated mean/SD (Student-t, n-1 d.f.) and
(b) the extra variance of a single future point (factor sqrt(1 + 1/n)).
Measured false-positive rate on pure noise (seed=1234, 100k trials/row):
n=3 22.7%, n=4 16.9%, n=5 14.4%, n=8 10.3%, n=15 7.1%, n=30 5.8% (nominal 5%).

The decision rule is now the two-sided 95% prediction interval for one new
observation:  |x - mean| / (s * sqrt(1 + 1/n)) > t_{0.975, n-1}.
Under iid Gaussian readings this has ~5% false-positive rate at every n
(verified by seeded simulation in tests/test_personalized_anomaly.py). Real
patient data are autocorrelated/non-Gaussian, so this is a screening aid
with an honest nominal error rate, not a validated change-point detector.
"""
from __future__ import annotations

import math
import statistics

MIN_READINGS_FOR_STATS = 3
CONFIDENCE_LEVEL = 0.95
# Relative tolerance below which a deviation from a perfectly constant
# history is treated as rounding noise (heuristic, documented).
ZERO_VARIANCE_REL_TOL = 0.05
ZERO_VARIANCE_ABS_TOL = 1e-6
# A sample SD below this (relative to max(1, |mean|)) is floating-point representation noise, not a measured spread:
# 0.1+0.2 vs 0.3 gives SD 2.8e-17, which made z = 3.6e9 and flagged a 3e-7 relative deviation (Phase 3, D07).
# Same guard as the Q-statistic code in app/ml/longitudinal.py.
SD_NOISE_FLOOR_REL = 1e-12

# Two-sided 95% Student-t critical values, d.f. 1..30 (standard table).
_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def t_critical_95(df: int) -> float:
    if df in _T_975:
        return _T_975[df]
    # df > 30: Cornish-Fisher style approximation, accurate to ~3 decimals.
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def t_critical(df: int, alpha: float = 0.05) -> float:
    """Two-sided Student-t critical value t_{1-alpha/2, df}. alpha=0.05 uses the standard table (reference-tested);
    other alphas use scipy's exact quantile."""
    if abs(alpha - 0.05) < 1e-12:
        return t_critical_95(df)
    from scipy.stats import t as _t
    return float(_t.ppf(1 - alpha / 2, df))


def effective_z_threshold(n: int, alpha: float = 0.05) -> float:
    """|z| (z = (x-mean)/s) above which a new point lies outside the (1-alpha) prediction interval."""
    return t_critical(n - 1, alpha) * math.sqrt(1 + 1 / n)


def _result(method, n, mean=None, stdev=None, z=None, anomaly=False, threshold=None, note=None):
    out = {
        "method": method,
        "readings_available": n,
        "readings_required": MIN_READINGS_FOR_STATS,
        "mean": mean,
        "stdev": stdev,
        "z_score": z,
        "is_anomaly": anomaly,
        "z_threshold_used": threshold,
    }
    if note:
        out["note"] = note
    return out


def detect_personal_anomaly(
    historical_values: list[float],
    latest_value: float,
    z_threshold: float | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compare `latest_value` against the patient's OWN prior readings.

    method: "z_score" | "zero_variance" | "insufficient_history" | "invalid_input"
    `z_threshold=None` (default) uses the exact (1-alpha) prediction-interval
    threshold for the available history length (alpha=0.05 by default); pass a number to override.
    NOTE (Phase 2): applied at EVERY new reading, a 5% per-reading rate gives ~80% family-wise false alarms over 31
    successive reviews (ml/simulation/results/anomaly_study.json). The longitudinal analysis therefore raises a SIGNAL
    only at a much stricter alpha and reports the 5% result as an informational note.

    Never raises. Non-finite history values are dropped (and counted in
    `note`); a non-finite `latest_value` yields method "invalid_input".
    """
    try:
        latest = float(latest_value)
    except (TypeError, ValueError):
        return _result("invalid_input", 0, note="latest value is not numeric")
    if not math.isfinite(latest):
        return _result("invalid_input", 0, note="latest value is not finite")

    clean: list[float] = []
    dropped = 0
    for v in historical_values or []:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if math.isfinite(fv):
            clean.append(fv)
        else:
            dropped += 1
    note = f"{dropped} non-finite/non-numeric history value(s) ignored" if dropped else None

    n = len(clean)
    if n < MIN_READINGS_FOR_STATS:
        return _result("insufficient_history", n, note=note)

    mean = statistics.fmean(clean)
    stdev = statistics.stdev(clean)  # sample SD (n-1)

    if stdev <= SD_NOISE_FLOOR_REL * max(1.0, abs(mean)):
        tol = max(ZERO_VARIANCE_ABS_TOL, ZERO_VARIANCE_REL_TOL * abs(mean))
        return _result("zero_variance", n, mean=mean, stdev=0.0, anomaly=abs(latest - mean) > tol, note=note)

    threshold = z_threshold if z_threshold is not None else effective_z_threshold(n, alpha)
    z = (latest - mean) / stdev
    return _result("z_score", n, mean=round(mean, 4), stdev=round(stdev, 4), z=round(z, 4),
                   anomaly=abs(z) >= threshold, threshold=round(threshold, 4), note=note)
