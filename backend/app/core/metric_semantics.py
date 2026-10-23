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

import json
import logging
import math
import re
from functools import lru_cache
from typing import Literal, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

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


STABLE_BAND = 0.05        # default; override with METRIC_STABLE_BAND
ADVERSE_THRESHOLD = 0.20  # default; override with METRIC_ADVERSE_THRESHOLD
# Both defaults are PROVISIONAL conventions, not clinical standards. A minimal clinically important difference (MCID)
# is used for a metric ONLY if it is listed in the file at METRIC_MCID_PATH with `approved_by` and `source` set;
# anything else is ignored (and says so), never silently applied. Format:
#   {"updrs_total": {"mcid": 3.0, "approved_by": "<clinician>", "source": "<protocol / citation>"}}


@lru_cache(maxsize=4)
def _load_mcid(path: str) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError("top level must be an object")
        return raw
    except Exception as exc:  # fail closed to the provisional rule
        logger.error("metric_mcid_file_unusable path=%s error_class=%s; using provisional thresholds", path, type(exc).__name__)
        return {}


def _approved_mcid(metric: str):
    """Returns (mcid, source, ignored_note). mcid is None unless clinician-approved and valid."""
    entry = _load_mcid(getattr(get_settings(), "metric_mcid_path", "") or "").get((metric or "").strip().lower())
    if entry is None:
        return None, None, ""
    ok = (isinstance(entry, dict) and isinstance(entry.get("mcid"), (int, float)) and not isinstance(entry.get("mcid"), bool)
          and math.isfinite(entry["mcid"]) and entry["mcid"] > 0
          and str(entry.get("approved_by") or "").strip() and str(entry.get("source") or "").strip())
    if not ok:
        return None, None, "mcid_ignored_not_clinician_approved"
    return float(entry["mcid"]), str(entry["source"]), ""


def _basis(extra: dict, *, mcid_source: Optional[str] = None) -> dict:
    if mcid_source:
        return {**extra, "threshold_basis": "clinician_approved_mcid", "threshold_provisional": False, "threshold_source": mcid_source}
    return {**extra, "threshold_basis": "provisional_relative_change", "threshold_provisional": True}


def classify_change(metric: str, baseline: Optional[float], current: Optional[float]) -> dict:
    """Polarity-aware classification of baseline -> current.

    Returns {classification, delta_abs, delta_pct (fraction or None), polarity, note, threshold_basis,
    threshold_provisional[, threshold_source]}.
    classification in: deteriorating | improving | stable | changed_direction_unknown |
                       baseline_zero_undefined | invalid | unknown
    """
    st = get_settings()
    stable_band = float(getattr(st, "metric_stable_band", STABLE_BAND))
    adverse_thr = float(getattr(st, "metric_adverse_threshold", ADVERSE_THRESHOLD))
    pol = metric_polarity(metric)
    mcid, mcid_src, ignored = _approved_mcid(metric)

    def out(classification, delta_abs, delta_pct, note="", use_mcid=False):
        n = "; ".join(x for x in (note, ignored) if x)
        return _basis({"classification": classification, "delta_abs": delta_abs, "delta_pct": delta_pct, "polarity": pol,
                       "note": n}, mcid_source=mcid_src if use_mcid else None)

    if baseline is None or current is None:
        return out("unknown", None, None, "missing baseline or current")
    if not (math.isfinite(baseline) and math.isfinite(current)):
        return out("invalid", None, None, "non-finite value")
    delta_abs = current - baseline
    if mcid is not None and pol != "unknown":
        # Clinician-approved absolute threshold replaces the provisional relative rule for this metric.
        adverse_abs = delta_abs if pol == "higher_worse" else -delta_abs
        dp = (delta_abs / abs(baseline)) if baseline != 0 else None
        cls = "deteriorating" if adverse_abs >= mcid else "improving" if adverse_abs <= -mcid else "stable"
        return out(cls, delta_abs, dp, use_mcid=True)
    if baseline == 0:
        if current == 0:
            return out("stable", 0.0, 0.0, "baseline and current are both 0")
        # Relative change is mathematically undefined; do NOT invent a percentage.
        return out("baseline_zero_undefined", delta_abs, None,
                   "baseline is 0 so relative change is undefined; absolute change reported instead")
    delta_pct = delta_abs / abs(baseline)
    # Round before threshold comparison: (1.2 - 1.0) / 1.0 == 0.19999999999999996 in IEEE-754, which would
    # otherwise miss an exactly-20% change.
    cmp_pct = round(delta_pct, 9)
    if abs(cmp_pct) < stable_band:
        return out("stable", delta_abs, delta_pct)
    if pol == "unknown":
        if abs(cmp_pct) >= adverse_thr:
            return out("changed_direction_unknown", delta_abs, delta_pct,
                       "metric polarity not registered; clinician must judge whether this change is adverse")
        return out("stable", delta_abs, delta_pct)
    adverse = cmp_pct if pol == "higher_worse" else -cmp_pct  # positive == moving in the bad direction
    if adverse >= adverse_thr:
        cls = "deteriorating"
    elif adverse <= -adverse_thr:
        cls = "improving"
    else:
        cls = "stable"
    return out(cls, delta_abs, delta_pct)
