"""
Metrics with explicit "undefined" handling (Phase 2, B2).

Rule: a metric is returned as `None` (never 0, never NaN, never a guess) when the data do not support it,
and the reason is recorded in the `undefined` map of the result. Example: ROC-AUC on a set that contains one
class only.

Resampling unit: when observations are clustered (several recordings per person) confidence intervals are
computed by resampling CLUSTERS (`cluster_bootstrap`), because treating rows as independent understates the
uncertainty.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# ----------------------------------------------------------------------------- basic helpers
def _check_binary(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError("y must contain only 0/1")
    return y.astype(int)


def _check_prob(p, n: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.shape != (n,):
        raise ValueError("p must have the same length as y")
    if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        raise ValueError("probabilities must be finite and within [0, 1]")
    return p


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion k/n (None when n == 0)."""
    if n <= 0:
        return None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ----------------------------------------------------------------------------- threshold metrics
def threshold_metrics(y, pred) -> dict:
    """Confusion-matrix based metrics. Each ratio is None when its denominator is 0."""
    y, pred = _check_binary(y), _check_binary(pred)
    if y.shape != pred.shape:
        raise ValueError("y and pred must have the same length")
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_pos, n_neg = tp + fn, tn + fp
    undefined: dict[str, str] = {}

    def ratio(name, num, den, why):
        if den == 0:
            undefined[name] = why
            return None
        return num / den

    sens = ratio("sensitivity", tp, n_pos, "no positive cases")
    spec = ratio("specificity", tn, n_neg, "no negative cases")
    prec = ratio("precision", tp, tp + fp, "no positive predictions")
    f1 = ratio("f1", 2 * tp, 2 * tp + fp + fn, "no positives and no positive predictions")
    bal = None if (sens is None or spec is None) else (sens + spec) / 2
    if bal is None:
        undefined["balanced_accuracy"] = "needs both classes present"
    return {
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n": len(y), "n_pos": n_pos, "n_neg": n_neg,
        "accuracy": ratio("accuracy", tp + tn, len(y), "empty"),
        "sensitivity": sens, "specificity": spec, "precision": prec, "recall": sens, "f1": f1,
        "balanced_accuracy": bal,
        "sensitivity_wilson95": wilson_interval(tp, n_pos), "specificity_wilson95": wilson_interval(tn, n_neg),
        "undefined": undefined,
    }


def ppv_npv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> dict:
    """Bayes' rule predictive values for a TARGET prevalence (not the study prevalence)."""
    for name, v in (("sensitivity", sensitivity), ("specificity", specificity)):
        if v is None or not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]")
    if not (0.0 < prevalence < 1.0):
        raise ValueError("prevalence must be in (0, 1)")
    tp, fp = sensitivity * prevalence, (1 - specificity) * (1 - prevalence)
    tn, fn = specificity * (1 - prevalence), (1 - sensitivity) * prevalence
    return {"prevalence": prevalence,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else None,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else None}


# ----------------------------------------------------------------------------- score metrics
def expected_calibration_error(y, p, bins: int = 10) -> float:
    """ECE with equal-width bins: sum_b (n_b/n) * |mean(y in b) - mean(p in b)|."""
    y = _check_binary(y)
    p = _check_prob(p, len(y))
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_table(y, p, bins: int = 10) -> list[dict]:
    y = _check_binary(y)
    p = _check_prob(p, len(y))
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return [{"bin": b, "n": int((idx == b).sum()),
             "mean_predicted": float(p[idx == b].mean()), "observed_rate": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def score_metrics(y, p, *, bins: int = 10) -> dict:
    """Threshold-free metrics. ROC-AUC / PR-AUC / calibration are None unless both classes are present."""
    y = _check_binary(y)
    p = _check_prob(p, len(y))
    undefined: dict[str, str] = {}
    both = len(np.unique(y)) == 2
    out = {"n": len(y), "prevalence": float(y.mean()) if len(y) else None}
    if both:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        out["pr_auc_chance_level"] = float(y.mean())  # PR-AUC of a no-skill classifier = prevalence
    else:
        out["roc_auc"] = out["pr_auc"] = out["pr_auc_chance_level"] = None
        undefined["roc_auc"] = undefined["pr_auc"] = "only one class present"
    out["brier"] = float(brier_score_loss(y, p)) if len(y) else None
    out["brier_constant_prevalence_baseline"] = (float(brier_score_loss(y, np.full(len(y), y.mean())))
                                                 if len(y) else None)
    out["ece"] = expected_calibration_error(y, p, bins) if len(y) else None
    out["undefined"] = undefined
    return out


def evaluate_binary(y, p, threshold: float = 0.5) -> dict:
    """Score metrics + threshold metrics at a threshold that MUST have been fixed before looking at this data
    (a-priori value, or chosen on training folds only)."""
    y = _check_binary(y)
    p = _check_prob(p, len(y))
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    return {"threshold": threshold, **score_metrics(y, p),
            "at_threshold": threshold_metrics(y, (p >= threshold).astype(int))}


# ----------------------------------------------------------------------------- regression
def regression_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if y.shape != yhat.shape or y.ndim != 1:
        raise ValueError("y and yhat must be 1-D and equal length")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(yhat))):
        raise ValueError("non-finite values")
    n = len(y)
    if n == 0:
        raise ValueError("empty input")
    err = yhat - y
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    out = {"n": n, "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
           "residual_mean": float(err.mean()), "residual_sd": float(err.std(ddof=1)) if n > 1 else None,
           "undefined": {}}
    if ss_tot == 0:
        out["undefined"]["r2"] = "target has zero variance"
    return out


# ----------------------------------------------------------------------------- aggregation + bootstrap
def aggregate_by_group(groups, y, p, how: str = "mean") -> pd.DataFrame:
    """One row per group: label (must be constant within the group) and the mean/median predicted probability."""
    df = pd.DataFrame({"group": np.asarray(groups), "y": np.asarray(y), "p": np.asarray(p, float)})
    if (df.groupby("group")["y"].nunique() > 1).any():
        raise ValueError("labels vary within a group; subject-level aggregation of a row-level target is invalid")
    agg = {"mean": "mean", "median": "median"}[how]
    return df.groupby("group").agg(y=("y", "first"), p=("p", agg)).reset_index()


def cluster_bootstrap(df: pd.DataFrame, cluster: str, stat: Callable[[pd.DataFrame], float | None],
                      *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI by resampling whole clusters with replacement. Resamples where `stat` is undefined
    (returns None/NaN, e.g. single class) are discarded and COUNTED; if too few survive, no interval is given."""
    rng = np.random.default_rng(seed)
    parts = {k: g for k, g in df.groupby(cluster, sort=True)}
    keys = np.array(list(parts.keys()), dtype=object)
    point = stat(df)
    vals: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        v = stat(pd.concat([parts[keys[i]] for i in pick], ignore_index=True))
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))
    out = {"point": None if point is None else float(point), "n_boot": n_boot, "n_valid": len(vals),
           "n_discarded": n_boot - len(vals), "n_clusters": len(keys), "alpha": alpha}
    if len(vals) >= max(20, 0.5 * n_boot):
        out["lo"] = float(np.percentile(vals, 100 * alpha / 2))
        out["hi"] = float(np.percentile(vals, 100 * (1 - alpha / 2)))
        if n_boot < 200:
            out["note"] = "n_boot < 200: percentile interval is coarse; use >= 1000 for reporting"
    else:
        out["lo"] = out["hi"] = None
        out["note"] = "too many degenerate resamples for a reliable interval"
    return out


def auc_stat(df: pd.DataFrame, y="y", p="p"):
    return None if df[y].nunique() < 2 else float(roc_auc_score(df[y], df[p]))


def paired_cluster_bootstrap_diff(df: pd.DataFrame, cluster: str, p_a: str, p_b: str, y: str = "y",
                                  *, n_boot: int = 1000, seed: int = 0) -> dict:
    """CI for AUC(a) - AUC(b) on the SAME clusters (paired). If the interval contains 0 the data cannot
    distinguish the two predictors."""
    return cluster_bootstrap(
        df, cluster,
        lambda d: (None if d[y].nunique() < 2 else float(roc_auc_score(d[y], d[p_a]) - roc_auc_score(d[y], d[p_b]))),
        n_boot=n_boot, seed=seed)


def group_label_permutation_test(groups, y, p_fn: Callable[[np.ndarray], np.ndarray], observed: float,
                                 *, n_perm: int = 200, seed: int = 0, stat=None) -> dict:
    """Permutation test for a GROUP-LEVEL label (e.g. PD vs healthy per person): shuffle the label among
    groups (never among rows), refit/evaluate via `p_fn(y_perm) -> out-of-fold probabilities`, compare.
    Valid only when the label is constant within each group."""
    g = np.asarray(groups)
    y = np.asarray(y)
    lab = pd.Series(y).groupby(g).agg(["nunique", "first"])
    if (lab["nunique"] > 1).any():
        raise ValueError("label varies within a group: group-level permutation is not valid for this target")
    stat = stat or (lambda yy, pp: None if len(np.unique(yy)) < 2 else float(roc_auc_score(yy, pp)))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        sh = pd.Series(rng.permutation(lab["first"].to_numpy()), index=lab.index)
        yp = pd.Series(g).map(sh).to_numpy().astype(int)
        v = stat(yp, p_fn(yp))
        if v is not None:
            null.append(v)
    if not null:
        return {"observed": observed, "n_perm": n_perm, "n_valid": 0, "p_value": None}
    null = np.array(null)
    return {"observed": observed, "n_perm": n_perm, "n_valid": int(len(null)), "null_mean": float(null.mean()),
            "null_95th": float(np.percentile(null, 95)),
            "p_value": float((1 + np.sum(null >= observed)) / (1 + len(null)))}
