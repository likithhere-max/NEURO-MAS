import math
import random

import pytest

from app.ml.personalized_anomaly import (
    MIN_READINGS_FOR_STATS, detect_personal_anomaly, effective_z_threshold, t_critical_95,
)


def test_t_critical_matches_standard_table():
    # Textbook two-sided 95% Student-t values.
    assert t_critical_95(2) == pytest.approx(4.303, abs=1e-3)
    assert t_critical_95(9) == pytest.approx(2.262, abs=1e-3)
    assert t_critical_95(30) == pytest.approx(2.042, abs=1e-3)
    assert t_critical_95(60) == pytest.approx(2.000, abs=2e-3)   # approximation branch
    assert t_critical_95(1000) == pytest.approx(1.962, abs=2e-3)


def test_effective_threshold_hand_calculation():
    # n=5: t(4)=2.776, sqrt(1+1/5)=1.095445 -> 3.0410
    assert effective_z_threshold(5) == pytest.approx(2.776 * math.sqrt(1.2), abs=1e-6)


def test_hand_calculated_z_score():
    hist = [10.0, 12.0, 11.0, 13.0, 9.0]       # mean 11, sample sd sqrt(10/4)=1.5811
    r = detect_personal_anomaly(hist, 20.0)
    assert r["mean"] == pytest.approx(11.0)
    assert r["stdev"] == pytest.approx(math.sqrt(2.5), abs=1e-4)
    assert r["z_score"] == pytest.approx((20 - 11) / math.sqrt(2.5), abs=1e-3)  # 5.69
    assert r["is_anomaly"] is True
    assert detect_personal_anomaly(hist, 13.5)["is_anomaly"] is False          # z=1.58 < 3.04


def test_old_threshold_would_have_flagged_this_ordinary_point():
    hist = [10.0, 12.0, 11.0, 13.0, 9.0]
    r = detect_personal_anomaly(hist, 14.5)     # z = 2.21: old rule (>=2) flagged it; prediction interval does not
    assert r["z_score"] == pytest.approx(2.214, abs=1e-3)
    assert r["is_anomaly"] is False


def test_insufficient_history():
    r = detect_personal_anomaly([1.0, 2.0], 5.0)
    assert r["method"] == "insufficient_history" and r["is_anomaly"] is False
    assert r["readings_required"] == MIN_READINGS_FOR_STATS


def test_zero_variance_tolerance():
    assert detect_personal_anomaly([5.0, 5.0, 5.0], 5.001)["is_anomaly"] is False   # rounding noise
    assert detect_personal_anomaly([5.0, 5.0, 5.0], 6.0)["is_anomaly"] is True       # 20% jump
    assert detect_personal_anomaly([0.0, 0.0, 0.0], 0.5)["is_anomaly"] is True


def test_nan_and_inf_never_crash_or_flag():
    assert detect_personal_anomaly([1, 1.1, 0.9], float("nan"))["method"] == "invalid_input"
    assert detect_personal_anomaly([1, 1.1, 0.9], float("inf"))["method"] == "invalid_input"
    r = detect_personal_anomaly([1.0, float("nan"), 1.2, 1.1, 1.05], 50.0)    # NaN dropped, rest used
    assert r["method"] == "z_score" and r["is_anomaly"] is True and "ignored" in r["note"]
    assert detect_personal_anomaly([float("nan")] * 5, 1.0)["method"] == "insufficient_history"


@pytest.mark.parametrize("n", [3, 5, 10, 30])
def test_false_positive_rate_is_nominal_on_pure_noise(n):
    """Seeded Monte-Carlo: under iid Gaussian noise the flag rate must be ~5% at EVERY history length.
    (The previous |z|>=2 rule measured 22.7% / 14.4% / 8-10% / 5.8% at n=3/5/10/30.)"""
    rng = random.Random(20261023 + n)
    trials = 20000
    fp = sum(
        detect_personal_anomaly([rng.gauss(0, 1) for _ in range(n)], rng.gauss(0, 1))["is_anomaly"]
        for _ in range(trials)
    )
    rate = fp / trials
    assert 0.040 <= rate <= 0.060, rate     # 5% +- 1pp (binomial SE ~0.15pp)


def test_detects_a_real_shift_with_reasonable_power():
    rng = random.Random(7)
    trials = 4000
    hits = sum(
        detect_personal_anomaly([rng.gauss(0, 1) for _ in range(10)], rng.gauss(4, 1))["is_anomaly"]
        for _ in range(trials)
    )
    assert hits / trials > 0.85
