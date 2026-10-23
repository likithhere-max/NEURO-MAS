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

# Phase 3 (D03): the original registry matched SUBSTRINGS of the name, which mis-assigned polarity ("stride_time_variability"
# -> lower_worse because it contains "stride"; "missteps_count" -> lower_worse because it contains "steps";
# "seizure_free_days" -> higher_worse because it contains "seizure"). Matching is now on whole tokens / adjacent token
# phrases, a name carrying a modifier that changes or blurs the direction is "unknown", and a name that matches BOTH
# families is "unknown". "unknown" is always safe: the change is reported for a clinician to judge, never guessed.
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_HIGHER_STEMS = ("tremor", "seizure", "updrs", "rigid", "bradykin", "dyskin", "pain", "headache", "fatigue", "severity",
                 "relapse", "edss", "hoehn", "freezing")                       # token START (inflections allowed)
_HIGHER_WORDS = {"fall", "falls"}                                              # whole token only
_HIGHER_PHRASES = ("symptom_score", "missed_dose", "sleep_latency")            # adjacent whole tokens
_LOWER_STEMS = ("cognit", "grip", "strength", "balance", "adherence", "memory")
_LOWER_WORDS = {"moca", "mmse", "fvc", "steps"}
_LOWER_PHRASES = ("gait_speed", "walking_speed", "walk_speed", "stride_length", "step_length", "sleep_hours",
                  "sleep_duration", "sleep_efficiency", "verbal_fluency")
# Tokens that invert the meaning ("free"), measure spread rather than level, or describe a change/impairment whose
# direction the registry does not encode.
_DIRECTION_MODIFIERS = {"variability", "variance", "cv", "sd", "std", "asymmetry", "free", "impairment", "complaint",
                        "complaints", "loss", "decline", "improvement", "reduction", "change", "delta"}


def metric_polarity(metric: str) -> Polarity:
    toks = [t for t in _TOKEN_SPLIT.split((metric or "").strip().lower()) if t]
    if not toks or _DIRECTION_MODIFIERS & set(toks):
        return "unknown"
    joined = "_" + "_".join(toks) + "_"
    higher = (any(t.startswith(st) for t in toks for st in _HIGHER_STEMS) or bool(_HIGHER_WORDS & set(toks))
              or any(f"_{ph}_" in joined for ph in _HIGHER_PHRASES))
    lower = (any(t.startswith(st) for t in toks for st in _LOWER_STEMS) or bool(_LOWER_WORDS & set(toks))
             or any(f"_{ph}_" in joined for ph in _LOWER_PHRASES))
    if higher and lower:
        return "unknown"
    return "higher_worse" if higher else "lower_worse" if lower else "unknown"


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
        # Phase 3 (D02): `current - baseline` carries IEEE-754 representation error (1.4 - 1.1 = 0.2999999999999998), so a change
        # equal to the MCID up to rounding must still count. Tolerance = 1e-9 relative to the MCID plus 8 ulps of the operands.
        tol = 1e-9 * mcid + 8 * 2.220446049250313e-16 * max(abs(baseline), abs(current))
        cls = "deteriorating" if adverse_abs >= mcid - tol else "improving" if adverse_abs <= -mcid + tol else "stable"
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
