"""
Metric polarity ("which direction is worse?") and change classification.

Audit finding (Critical/High): the longitudinal agent and
repository.build_patient_state treated ANY increase >= 20% as
"deteriorating" and ANY decrease as "improving" for EVERY metric. A fall in
gait speed (1.2 -> 0.7 m/s) or MoCA score (28 -> 18) was labelled "improving"
and the personalization agent then advised "advance rehabilitation".

This module is the single place that knows which direction is adverse for a
metric. Unknown metrics are NEVER classified as improving/deteriorating --
they are reported as "changed_direction_unknown" so a clinician decides.

The registry is a curated, conservative convention list, not a clinical
standard; extend it deliberately.
"""
from __future__ import annotations

import math
import re
from typing import Literal, Optional

Polarity = Literal["higher_worse", "lower_worse", "unknown"]

_HIGHER_WORSE_PATTERNS = [
    r"tremor", r"seizure", r"fall(s|_count|_rate)?\b", r"updrs", r"rigid", r"bradykin", r"dyskin",
    r"pain", r"headache", r"fatigue", r"severity", r"symptom_score", r"relapse", r"missed_dose",
    r"edss", r"hoehn", r"sleep_latency", r"freezing",
]
_LOWER_WORSE_PATTERNS = [
    r"gait_speed", r"walking_speed", r"walk_speed", r"stride", r"step_length", r"moca", r"mmse",
    r"cognit", r"grip", r"strength", r"balance_score", r"balance", r"adherence", r"steps", r"fvc",
    r"sleep_hours", r"sleep_duration", r"sleep_efficiency", r"verbal_fluency", r"memory",
]


def metric_polarity(metric: str) -> Polarity:
    m = (metric or "").strip().lower()
    # Order matters only on overlap; "adherence"/"steps" are lower-worse, "fatigue" higher-worse.
    for pat in _LOWER_WORSE_PATTERNS:
        if re.search(pat, m):
            return "lower_worse"
    for pat in _HIGHER_WORSE_PATTERNS:
        if re.search(pat, m):
            return "higher_worse"
    return "unknown"


STABLE_BAND = 0.05        # |relative change| below this is "stable"
ADVERSE_THRESHOLD = 0.20  # relative change at/above this in the adverse direction is flagged


def classify_change(metric: str, baseline: Optional[float], current: Optional[float]) -> dict:
    """Polarity-aware classification of baseline -> current.

    Returns {classification, delta_abs, delta_pct (fraction or None), polarity, note}.
    classification in: deteriorating | improving | stable | changed_direction_unknown |
                       baseline_zero_undefined | invalid | unknown
    """
    pol = metric_polarity(metric)
    if baseline is None or current is None:
        return {"classification": "unknown", "delta_abs": None, "delta_pct": None, "polarity": pol, "note": "missing baseline or current"}
    if not (math.isfinite(baseline) and math.isfinite(current)):
        return {"classification": "invalid", "delta_abs": None, "delta_pct": None, "polarity": pol, "note": "non-finite value"}
    delta_abs = current - baseline
    if baseline == 0:
        if current == 0:
            return {"classification": "stable", "delta_abs": 0.0, "delta_pct": 0.0, "polarity": pol, "note": "baseline and current are both 0"}
        # Relative change is mathematically undefined; do NOT invent a percentage.
        return {"classification": "baseline_zero_undefined", "delta_abs": delta_abs, "delta_pct": None, "polarity": pol,
                "note": "baseline is 0 so relative change is undefined; absolute change reported instead"}
    delta_pct = delta_abs / abs(baseline)
    # Round before threshold comparison: (1.2 - 1.0) / 1.0 == 0.19999999999999996 in IEEE-754, which would
    # otherwise miss an exactly-20% change.
    cmp_pct = round(delta_pct, 9)
    if abs(cmp_pct) < STABLE_BAND:
        return {"classification": "stable", "delta_abs": delta_abs, "delta_pct": delta_pct, "polarity": pol, "note": ""}
    if pol == "unknown":
        if abs(cmp_pct) >= ADVERSE_THRESHOLD:
            return {"classification": "changed_direction_unknown", "delta_abs": delta_abs, "delta_pct": delta_pct, "polarity": pol,
                    "note": "metric polarity not registered; clinician must judge whether this change is adverse"}
        return {"classification": "stable", "delta_abs": delta_abs, "delta_pct": delta_pct, "polarity": pol, "note": ""}
    adverse = cmp_pct if pol == "higher_worse" else -cmp_pct  # positive == moving in the bad direction
    if adverse >= ADVERSE_THRESHOLD:
        cls = "deteriorating"
    elif adverse <= -ADVERSE_THRESHOLD:
        cls = "improving"
    else:
        cls = "stable"
    return {"classification": cls, "delta_abs": delta_abs, "delta_pct": delta_pct, "polarity": pol, "note": ""}
