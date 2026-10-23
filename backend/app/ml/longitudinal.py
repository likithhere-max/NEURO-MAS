"""
Longitudinal analysis of ONE patient's time-stamped measurements for ONE metric (Phase 2, C2/C3).

Questions are separated, because each needs a different tool and a different error rate:

  Q1  "Is the newest reading unusual vs this patient's own history?"   -> Student-t prediction interval (existing
                                                                          detector) + robust z (median/MAD), report-only
  Q2  "Has the level shifted and stayed shifted?"                      -> tabular CUSUM and EWMA on self-starting
                                                                          (Quesenberry Q) statistics; suppressed when the series
                                                                          is too short or positively autocorrelated
  Q3  "Is there a monotone drift over time?"                           -> Theil-Sen slope (robust to outliers, valid for
                                                                          irregular times) + Mann-Kendall test with a lag-1
                                                                          autocorrelation variance correction

Methods that the data cannot support are NOT run; they appear in the report with status "not_assessed" and the
reason (too few readings, no baseline, zero span...). Nothing is imputed.

Reading the output (see `AnomalyReport`): a SIGNAL means "a pre-specified statistical rule fired on this
patient's own series; a clinician should look". It is never a diagnosis and never a probability of disease.
A signal is also not proof of clinically meaningful change: `clinical_significance` stays "not assessed" unless a
clinician-approved minimal important difference is configured for the metric (`mcid`).

All thresholds are PROVISIONAL engineering parameters with a provenance string; they are overridable through
`AnalysisConfig`. Their nominal error rates assume iid Gaussian noise. `ml/simulation/anomaly_study.py` measures
what actually happens under autocorrelation and under repeated testing, and docs/phase2/MATHEMATICAL_VALIDATION.md
reports it.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np

from app.core.metric_semantics import metric_polarity
from app.ml.personalized_anomaly import MIN_READINGS_FOR_STATS, detect_personal_anomaly

METHOD_VERSION = "longitudinal-1.0"
MAX_ABS_VALUE = 1e9  # readings beyond this magnitude are treated as invalid (overflow / unit-error guard)
MAD_TO_SIGMA = 1.4826022185056018  # 1 / Phi^{-1}(0.75): makes MAD consistent for the Gaussian SD


# ----------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class AnalysisConfig:
    # --- baseline
    baseline_n: int = 5                 # first k readings define the REPORTED within-person baseline (provisional)
    # --- Q1 point test. A single review at alpha=0.05 is fine, but the rule is re-applied at EVERY new reading:
    #     the simulation study measured ~80% family-wise false alarms over 31 reviews at alpha=0.05. So the SIGNAL level
    #     is far stricter (about 3% family-wise) and the 5% result is reported as an informational note only.
    point_alpha_signal: float = 0.001
    point_alpha_note: float = 0.05
    point_alpha_provenance: str = "Student-t prediction interval; signal alpha chosen from ml/simulation/results/anomaly_study.json (repeated testing)"
    # --- Q2 sequential, SELF-STARTING (Quesenberry Q-statistics). A fixed small baseline was evaluated and
    #     rejected: with 5-12 baseline readings textbook limits gave 30-65% false alarms over 35 steps even for iid
    #     Gaussian noise (ml/simulation/results/anomaly_study.json, section "rejected_fixed_baseline").
    sequential_min_n: int = 15          # need enough readings to estimate lag-1 autocorrelation at all
    sequential_max_rho: float = 0.3     # pre-alarm lag-1 autocorrelation above this -> alarm suppressed (error rate untrustworthy)
    cusum_k: float = 0.5                # reference value in SD units (tuned for ~1 SD shifts; Montgomery, SPC)
    cusum_h: float = 5.0                # decision interval (nominal ARL0 ~465 for iid N(0,1) inputs; here ~7% over 40 readings)
    ewma_lambda: float = 0.2            # smoothing (Montgomery recommends 0.05-0.25)
    ewma_L: float = 3.0                 # limit width in EWMA SDs
    # --- Q3 trend
    trend_min_n: int = 8                # Mann-Kendall is unreliable below ~8-10 points
    trend_alpha: float = 0.01           # 0.05 gave 27-55% family-wise false alarms over repeated reviews (study)
    trend_min_effect_sd: float = 1.0    # |slope| x span must be >= this many baseline SDs to be flagged (provisional)
    # --- sampling quality
    irregular_gap_cv: float = 0.75      # CV of inter-reading gaps above which sampling is called irregular
    stale_days: float | None = None     # warn if newest reading older than this (None: don't check)
    # --- clinical significance (None = not configured -> never claimed)
    mcid: float | None = None
    mcid_provenance: str | None = None


DEFAULT_CONFIG = AnalysisConfig()


# ----------------------------------------------------------------------------- input normalisation
def _to_days(t: Any) -> float:
    """float days (any origin), datetime, or ISO-8601 string -> float days since the Unix epoch."""
    if isinstance(t, (int, float, np.integer, np.floating)) and not isinstance(t, bool):
        v = float(t)
        if not math.isfinite(v):
            raise ValueError("non-finite time")
        return v
    if isinstance(t, str):
        t = datetime.fromisoformat(t.strip().replace("Z", "+00:00"))
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp() / 86400.0
    raise ValueError(f"unsupported time type {type(t).__name__}")


def clean_series(times: Sequence[Any], values: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Coerce, drop invalid pairs, sort by time, and merge exact-duplicate timestamps (median of repeats)."""
    if len(times) != len(values):
        raise ValueError("times and values must have the same length")
    pairs, dropped = [], 0
    for t, v in zip(times, values):
        try:
            tv, vv = _to_days(t), float(v)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if isinstance(v, bool) or not math.isfinite(vv) or abs(vv) > MAX_ABS_VALUE:
            dropped += 1
            continue
        pairs.append((tv, vv))
    pairs.sort(key=lambda p: p[0])
    merged: list[tuple[float, float]] = []
    repeats = 0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and abs(pairs[j + 1][0] - pairs[i][0]) < 1.0 / 1440:  # same minute
            j += 1
        grp = [p[1] for p in pairs[i:j + 1]]
        if len(grp) > 1:
            repeats += len(grp) - 1
        merged.append((pairs[i][0], float(statistics.median(grp))))
        i = j + 1
    t = np.array([m[0] for m in merged], float)
    y = np.array([m[1] for m in merged], float)
    return t, y, {"dropped_invalid": dropped, "merged_duplicate_timestamps": repeats}


def sampling_summary(t: np.ndarray, cfg: AnalysisConfig) -> dict:
    n = len(t)
    if n < 2:
        return {"n": n, "span_days": 0.0, "median_gap_days": None, "gap_cv": None, "irregular": None}
    gaps = np.diff(t)
    cv = float(np.std(gaps, ddof=1) / np.mean(gaps)) if n > 2 and np.mean(gaps) > 0 else None
    return {"n": n, "span_days": float(t[-1] - t[0]), "median_gap_days": float(np.median(gaps)),
            "max_gap_days": float(gaps.max()), "gap_cv": cv, "irregular": (cv is not None and cv > cfg.irregular_gap_cv)}


# ----------------------------------------------------------------------------- building blocks
def robust_location_scale(x: np.ndarray) -> tuple[float, float, str]:
    """median and MAD-based sigma; falls back to the sample SD when MAD == 0 (many ties); scale 0 -> reported."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 0:
        return med, MAD_TO_SIGMA * mad, "median/MAD"
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return med, sd, "sample_sd_fallback" if sd > 0 else "zero_variance"


def theil_sen(t: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """Median of pairwise slopes (Sen 1968); CI from scipy's implementation (Theil-Sen with Sen's intervals)."""
    from scipy.stats import theilslopes

    n = len(t)
    if n < 3 or np.ptp(t) == 0:
        return {"status": "not_assessed", "reason": "need >= 3 readings spanning > 0 days"}
    slope, intercept, lo, hi = theilslopes(y, t, alpha=1 - alpha)
    return {"status": "assessed", "slope_per_day": float(slope), "intercept": float(intercept),
            "slope_ci_lo": float(lo), "slope_ci_hi": float(hi), "n": n}


def mann_kendall(y: np.ndarray, *, autocorr_correct: bool = True, residuals: np.ndarray | None = None) -> dict:
    """Mann-Kendall trend test (normal approximation, tie-corrected variance) with optional lag-1 autocorrelation
    variance inflation  Var' = Var * (1 + rho)/(1 - rho)  for rho > 0  (rho from detrended residuals, capped at 0.9).
    Two-sided p. Valid for irregular spacing because it uses only the order of readings."""
    n = len(y)
    if n < 3:
        return {"status": "not_assessed", "reason": "need >= 3 readings"}
    s = 0
    for i in range(n - 1):
        s += int(np.sign(y[i + 1:] - y[i]).sum())
    _, counts = np.unique(y, return_counts=True)
    tie_term = float(sum(c * (c - 1) * (2 * c + 5) for c in counts if c > 1))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return {"status": "not_assessed", "reason": "all readings identical"}
    rho = None
    infl = 1.0
    if autocorr_correct and residuals is not None and len(residuals) >= 6 and np.std(residuals) > 0:
        r = residuals - residuals.mean()
        rho = float((r[:-1] * r[1:]).sum() / (r * r).sum())
        if rho > 0:
            rho_c = min(rho, 0.9)
            infl = (1 + rho_c) / (1 - rho_c)
    z = 0.0 if s == 0 else (s - np.sign(s)) / math.sqrt(var_s * infl)
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"status": "assessed", "S": int(s), "var_S": var_s, "variance_inflation": infl, "lag1_autocorr_residuals": rho,
            "z": float(z), "p_value": float(p), "n": n}


def cusum(z: Sequence[float], k: float, h: float) -> dict:
    """Two-sided tabular CUSUM on standardised deviations z_t:
         C+_t = max(0, C+_{t-1} + z_t - k)      C-_t = max(0, C-_{t-1} - z_t - k)
       alarm when C+ > h (upward shift) or C- > h (downward shift). Returns first alarm index (0-based) if any."""
    cp = cm = 0.0
    path_p, path_m = [], []
    first = None
    direction = None
    for i, v in enumerate(z):
        cp = max(0.0, cp + v - k)
        cm = max(0.0, cm - v - k)
        path_p.append(cp)
        path_m.append(cm)
        if first is None and (cp > h or cm > h):
            first, direction = i, ("up" if cp > h else "down")
    return {"C_plus": path_p, "C_minus": path_m, "k": k, "h": h, "first_alarm_index": first, "alarm_direction": direction,
            "final_C_plus": path_p[-1] if path_p else 0.0, "final_C_minus": path_m[-1] if path_m else 0.0}


def ewma(z: Sequence[float], lam: float, L: float) -> dict:
    """EWMA of standardised deviations: w_t = lam*z_t + (1-lam)*w_{t-1}, w_0 = 0.
       Time-varying limit  +/- L * sqrt(lam/(2-lam) * (1 - (1-lam)^{2t})).  Index-based (assumes roughly regular
       sampling; the report flags irregular sampling)."""
    w = 0.0
    path, limits = [], []
    first = None
    direction = None
    for t, v in enumerate(z, start=1):
        w = lam * v + (1 - lam) * w
        lim = L * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
        path.append(w)
        limits.append(lim)
        if first is None and abs(w) > lim:
            first, direction = t - 1, ("up" if w > 0 else "down")
    return {"path": path, "limits": limits, "lambda": lam, "L": L, "first_alarm_index": first, "alarm_direction": direction,
            "final_value": path[-1] if path else 0.0, "final_limit": limits[-1] if limits else None}


def q_statistics(x: Sequence[float], min_prior: int = 2) -> list[float | None]:
    """Quesenberry (1991) Q-statistics. For i > min_prior:
         r_i = sqrt(m/(m+1)) (x_i - mean_{1..m}) / s_{1..m},   m = i-1,   r_i ~ t_{m-1}  (iid Gaussian, in control)
         Q_i = Phi^{-1}( F_{t, m-1}(r_i) )                                              ~ N(0,1), mutually independent.
       These are exactly standard normal and independent under the iid Gaussian null REGARDLESS of how few earlier
       readings exist, which removes the estimation-error problem of fixed-baseline charts. The cost: a sustained
       shift is gradually absorbed into the running mean (sensitivity falls for old shifts).
       Entries for the first `min_prior` readings are None; a zero-variance history gives None (undefined)."""
    from scipy.stats import norm, t as tdist

    xs = np.asarray(x, float)
    out: list[float | None] = [None] * len(xs)
    for i in range(min_prior, len(xs)):
        m = i
        prior = xs[:m]
        sd = float(np.std(prior, ddof=1))
        if sd <= 1e-12 * max(1.0, abs(float(prior.mean()))):
            continue
        r = math.sqrt(m / (m + 1)) * (xs[i] - float(prior.mean())) / sd
        # use the tail that keeps precision, so that Q(-x) = -Q(x) exactly; |Q| is capped at 8 (p ~ 6e-16)
        tail = min(max(float(tdist.sf(abs(r), m - 1)), 1e-16), 0.5)
        q = float(norm.isf(tail))
        out[i] = q if r >= 0 else -q
    return out


def self_starting_sequential(x: Sequence[float], cfg: "AnalysisConfig") -> dict:
    q = q_statistics(x)
    z = [v for v in q if v is not None]
    cs = cusum(z, cfg.cusum_k, cfg.cusum_h)
    ew = ewma(z, cfg.ewma_lambda, cfg.ewma_L)
    # map alarm indices (in the z list) back to positions in x
    pos = [i for i, v in enumerate(q) if v is not None]
    for res in (cs, ew):
        fa = res["first_alarm_index"]
        res["first_alarm_reading_index"] = pos[fa] if fa is not None else None
    return {"n_q": len(z), "cusum": cs, "ewma": ew}


def prefix_lag1_autocorr(y: np.ndarray, upto: int, min_len: int = 10) -> float | None:
    """Lag-1 autocorrelation of y[:upto] (mean removed). Estimated on the PRE-ALARM prefix only, because a genuine level
    shift in the later readings looks exactly like strong autocorrelation (a detrended full-series estimate returned
    0.72 for an iid series with a 4-SD step and suppressed the very alarm it should confirm). Biased LOW for short
    prefixes (about -1/n) and noisy; None when the prefix is shorter than `min_len` (then the alarm cannot be vetted)."""
    if upto < min_len:
        return None
    r = np.asarray(y[:upto], float)
    r = r - r.mean()
    den = float((r * r).sum())
    return float((r[:-1] * r[1:]).sum() / den) if den > 0 else None


def lag1_autocorr_detrended(t: np.ndarray, y: np.ndarray) -> float | None:
    """Lag-1 autocorrelation of residuals from a Theil-Sen line (None if undefined). Biased low for short series."""
    if len(y) < 6 or np.ptp(t) == 0 or np.std(y) == 0:
        return None
    ts = theil_sen(t, y)
    r = y - (ts["intercept"] + ts["slope_per_day"] * t)
    r = r - r.mean()
    den = float((r * r).sum())
    return float((r[:-1] * r[1:]).sum() / den) if den > 0 else None


# ----------------------------------------------------------------------------- report
@dataclass
class AnomalyReport:
    metric: str
    detection_status: str
    method_version: str = METHOD_VERSION
    polarity: str = "unknown"
    baseline: dict = field(default_factory=dict)
    observation_count: int = 0
    sampling: dict = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    signals: list[dict] = field(default_factory=list)
    data_quality_warnings: list[str] = field(default_factory=list)
    clinical_significance: dict = field(default_factory=dict)
    uncertainty_and_limitations: list[str] = field(default_factory=list)
    is_diagnosis: bool = False

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @property
    def any_signal(self) -> bool:
        return bool(self.signals)


LIMITATIONS = [
    "Rules were designed for independent, roughly Gaussian noise; real clinical series are autocorrelated and "
    "state-dependent (e.g. medication ON/OFF), so false-alarm rates on real patients are not known.",
    "A signal means a statistical rule fired on this patient's own series, not that disease activity changed.",
    "Thresholds are provisional engineering parameters, not clinically validated cut-offs.",
    "Several methods are evaluated on the same data; their signals are correlated and are not independent confirmation.",
]


def _adverse(polarity: str, direction: str | None) -> bool | None:
    if direction is None or polarity == "unknown":
        return None
    return (direction == "up") == (polarity == "higher_worse")


def analyze_series(metric: str, times: Sequence[Any], values: Sequence[Any], config: AnalysisConfig = DEFAULT_CONFIG) -> AnomalyReport:
    pol = metric_polarity(metric)
    rep = AnomalyReport(metric=metric, detection_status="insufficient_data", polarity=pol,
                        uncertainty_and_limitations=list(LIMITATIONS))
    rep.clinical_significance = {"assessed": False, "mcid": config.mcid, "provenance": config.mcid_provenance,
                                 "reason": ("no clinician-approved minimal important difference configured for this metric"
                                            if config.mcid is None else "see observed_change_vs_mcid")}
    try:
        t, y, q = clean_series(times, values)
    except ValueError as e:
        rep.detection_status = "invalid_input"
        rep.data_quality_warnings.append(f"invalid_input: {e}")
        return rep
    n = len(y)
    rep.observation_count = n
    rep.sampling = sampling_summary(t, config)
    if q["dropped_invalid"]:
        rep.data_quality_warnings.append(f"{q['dropped_invalid']} reading(s) dropped (non-numeric, non-finite or bad timestamp)")
    if q["merged_duplicate_timestamps"]:
        rep.data_quality_warnings.append(f"{q['merged_duplicate_timestamps']} repeated reading(s) at the same time merged by median")
    if rep.sampling.get("irregular"):
        rep.data_quality_warnings.append("irregular_sampling: EWMA/CUSUM are index-based and less reliable; Theil-Sen/Mann-Kendall use real times/order")
    if pol == "unknown":
        rep.data_quality_warnings.append("metric_polarity_unknown: signals cannot be classified as adverse or favourable")
    if n < MIN_READINGS_FOR_STATS:
        rep.statistics["note"] = f"need >= {MIN_READINGS_FOR_STATS} valid readings; have {n}"
        return rep

    # ---- baseline (first k readings) -- explicit and reported
    b_n = min(config.baseline_n, n - 1) if n - 1 >= 2 else n
    base, mon = y[:b_n], y[b_n:]
    mu0, sigma0, scale_method = robust_location_scale(base)
    rep.baseline = {"definition": f"first {b_n} readings", "n": b_n, "start_day": float(t[0]), "end_day": float(t[b_n - 1]),
                    "location_median": mu0, "scale_robust_sigma": sigma0, "scale_method": scale_method}
    if sigma0 == 0:
        rep.data_quality_warnings.append("baseline_zero_variance: robust z not assessed")

    st: dict[str, Any] = {}
    th: dict[str, Any] = {}
    signals: list[dict] = []

    # ---- Q1 newest reading vs own history (prediction interval) ------------------------------
    latest, hist = float(y[-1]), [float(v) for v in y[:-1]]
    pt = detect_personal_anomaly(hist, latest, alpha=config.point_alpha_signal)
    note = detect_personal_anomaly(hist, latest, alpha=config.point_alpha_note)
    st["point_t_interval"] = {**{k: pt.get(k) for k in ("method", "z_score", "z_threshold_used", "is_anomaly", "readings_available", "mean", "stdev")},
                              "alpha": config.point_alpha_signal,
                              "unusual_at_note_level": {"alpha": config.point_alpha_note, "is_unusual": bool(note.get("is_anomaly")),
                                                        "z_threshold_used": note.get("z_threshold_used"),
                                                        "meaning": "informational only; at this alpha repeated review gives ~80% family-wise false alarms"}}
    th["point_t_interval"] = {"value": {"signal_alpha": config.point_alpha_signal, "z_threshold": pt.get("z_threshold_used")},
                              "provenance": config.point_alpha_provenance}
    if sigma0 > 0:
        st["robust_z_latest_vs_baseline"] = (latest - mu0) / sigma0
    if note.get("is_anomaly") and not pt.get("is_anomaly"):
        rep.data_quality_warnings.append("latest_reading_unusual_at_5pct_level_not_signalled: expected to occur by chance about once per 20 readings")
    if pt.get("is_anomaly"):
        d = "up" if latest > (pt.get("mean") if pt.get("mean") is not None else mu0) else "down"
        signals.append({"method": "t_prediction_interval", "kind": "point_outlier", "direction": d,
                        "adverse": _adverse(pol, d), "index": n - 1})

    # ---- Q2 sequential monitoring: SELF-STARTING Q-statistic CUSUM + EWMA -------------------------
    rho_full = lag1_autocorr_detrended(t, y)
    st["lag1_autocorrelation_detrended_full_series"] = rho_full
    if n < config.sequential_min_n:
        st["cusum"] = st["ewma"] = {"status": "not_assessed",
                                    "reason": f"need >= {config.sequential_min_n} readings (have {n})"}
    else:
        seq = self_starting_sequential(y, config)
        th["cusum"] = {"value": {"k": config.cusum_k, "h": config.cusum_h},
                       "provenance": "tabular CUSUM on self-starting Q-statistics; k,h standard SPC values (Montgomery); provisional"}
        th["ewma"] = {"value": {"lambda": config.ewma_lambda, "L": config.ewma_L},
                      "provenance": "EWMA on self-starting Q-statistics; standard SPC values (Montgomery); provisional"}
        th["sequential_reliability_gate"] = {
            "value": {"min_n": config.sequential_min_n, "max_prefix_lag1_rho": config.sequential_max_rho, "min_prefix_len": 10},
            "provenance": "an alarm is accepted only if the lag-1 autocorrelation of the PRE-ALARM readings is <= the cap; "
                          "chosen from ml/simulation/results/anomaly_study.json"}
        for name in ("cusum", "ewma"):
            res = seq[name]
            entry = {k: v for k, v in res.items() if k not in ("C_plus", "C_minus", "path", "limits")}
            fa = res["first_alarm_reading_index"]
            if fa is None:
                entry["status"] = "assessed"
                entry["prefix_lag1_rho"] = prefix_lag1_autocorr(y, n) 
            else:
                rho = prefix_lag1_autocorr(y, fa)
                entry["prefix_lag1_rho"] = rho
                if rho is not None and rho <= config.sequential_max_rho:
                    entry["status"] = "alarm_accepted"
                    d = res["alarm_direction"]
                    signals.append({"method": name, "kind": "level_shift", "direction": d, "adverse": _adverse(pol, d), "index": fa})
                else:
                    entry["status"] = "alarm_suppressed_autocorrelation_unvetted"
                    rep.data_quality_warnings.append(
                        f"{name}_alarm_suppressed: pre-alarm lag-1 autocorrelation is "
                        + ("not estimable (<10 pre-alarm readings)" if rho is None else f"{rho:.2f} > {config.sequential_max_rho:.2f}")
                        + "; nominal error rates do not hold, so no signal is raised")
            st[name] = entry

    # ---- Q3 trend -----------------------------------------------------------------------------
    if n >= config.trend_min_n and np.ptp(t) > 0:
        ts = theil_sen(t, y, config.trend_alpha)
        resid = y - (ts["intercept"] + ts["slope_per_day"] * t)
        mk = mann_kendall(y, residuals=resid)
        effect_sd = abs(ts["slope_per_day"]) * float(np.ptp(t)) / sigma0 if sigma0 > 0 else None
        st["theil_sen"], st["mann_kendall"] = ts, mk
        st["trend_total_change_over_span"] = ts["slope_per_day"] * float(np.ptp(t))
        st["trend_effect_in_baseline_sd"] = effect_sd
        th["trend"] = {"value": {"alpha": config.trend_alpha, "min_effect_baseline_sd": config.trend_min_effect_sd},
                       "provenance": "Mann-Kendall two-sided alpha (conventional) + provisional minimum-effect rule"}
        if (mk.get("status") == "assessed" and mk["p_value"] < config.trend_alpha
                and (effect_sd is None or effect_sd >= config.trend_min_effect_sd)):
            d = "up" if ts["slope_per_day"] > 0 else "down"
            signals.append({"method": "theil_sen_mann_kendall", "kind": "trend", "direction": d, "adverse": _adverse(pol, d), "index": None})
        if mk.get("lag1_autocorr_residuals") is not None and mk["lag1_autocorr_residuals"] > 0.3:
            rep.data_quality_warnings.append("positive_autocorrelation: trend p-value was inflated-variance corrected but remains approximate")
    else:
        st["theil_sen"] = st["mann_kendall"] = {"status": "not_assessed",
                                                "reason": f"need >= {config.trend_min_n} readings spanning > 0 days (have {n})"}

    if config.mcid is not None:
        change = abs(latest - mu0)
        rep.clinical_significance.update({"assessed": True, "observed_change_vs_mcid": change / config.mcid,
                                          "meets_mcid": change >= config.mcid, "reason": "latest vs baseline median compared to configured MCID"})

    rep.statistics, rep.thresholds, rep.signals = st, th, signals
    if not signals:
        rep.detection_status = "no_signal"
    else:
        adv = [s["adverse"] for s in signals]
        rep.detection_status = ("signal_adverse" if any(a is True for a in adv) else
                                "signal_favourable" if all(a is False for a in adv) else "signal_direction_unknown")
    if n < 8:
        rep.data_quality_warnings.append("short_series: low power; absence of a signal is weak evidence of stability")
    return rep
