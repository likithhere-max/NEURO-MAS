"""
Personalized within-patient longitudinal anomaly detection (Roadmap item 6,
Track 1). Pure statistics on ONE patient's own historical readings for a
single metric -- this is deliberately NOT a trained model and NOT a
population classifier. It answers "is this reading unusual for THIS
person specifically," using that person's own mean and standard deviation
as the reference point -- which is honest even with a tiny amount of data,
unlike a population-level diagnostic model would be (see
claude/neuro-mas-architecture-v2.md, "ML" section).
"""
from __future__ import annotations

import statistics

MIN_READINGS_FOR_STATS = 3
DEFAULT_Z_THRESHOLD = 2.0


def detect_personal_anomaly(
    historical_values: list[float],
    latest_value: float,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> dict:
    """
    Compares `latest_value` against the patient's OWN historical readings for
    one metric (never against other patients). Returns a dict with:
      method              "z_score" | "zero_variance" | "insufficient_history"
      readings_available  how many prior readings were available
      readings_required   the minimum needed for a real z-score
      mean / stdev        the patient's own baseline stats (None if insufficient)
      z_score             how many of the patient's own std devs away the
                           latest reading is (None if insufficient or zero variance)
      is_anomaly          bool

    Never raises. Insufficient history and zero variance both degrade
    gracefully to an honest "not enough information" result rather than a
    fabricated flag or a crash.
    """
    n = len(historical_values)
    if n < MIN_READINGS_FOR_STATS:
        return {
            "method": "insufficient_history",
            "readings_available": n,
            "readings_required": MIN_READINGS_FOR_STATS,
            "mean": None,
            "stdev": None,
            "z_score": None,
            "is_anomaly": False,
        }

    mean = statistics.mean(historical_values)
    stdev = statistics.stdev(historical_values)  # sample stdev (n-1 denominator)

    if stdev == 0:
        # Every historical reading was identical -- a z-score is undefined
        # (division by zero), but any deviation at all is still meaningful.
        return {
            "method": "zero_variance",
            "readings_available": n,
            "readings_required": MIN_READINGS_FOR_STATS,
            "mean": mean,
            "stdev": 0.0,
            "z_score": None,
            "is_anomaly": latest_value != mean,
        }

    z_score = (latest_value - mean) / stdev
    return {
        "method": "z_score",
        "readings_available": n,
        "readings_required": MIN_READINGS_FOR_STATS,
        "mean": round(mean, 4),
        "stdev": round(stdev, 4),
        "z_score": round(z_score, 4),
        "is_anomaly": abs(z_score) >= z_threshold,
    }
    