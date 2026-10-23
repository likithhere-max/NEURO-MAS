"""
Comprehensive binary-endpoint evaluation (Phase 6, Workstream 3).

What this adds to `metrics.py`: log loss, calibration intercept and slope, decision-curve net benefit, a single cluster
bootstrap that yields intervals for EVERY reported metric from the same resamples (so paired comparisons are paired),
subgroup reporting with fixed minimum-size rules, and grouped cross-fitted recalibration.

Rules kept from `metrics.py`:
  * a metric that the data do not support is returned as None with a reason; it is never 0, NaN or a guess;
  * the resampling unit is the CLUSTER (subject), never the row;
  * thresholds are inputs.  This module never chooses an operating point from the data it evaluates.

Nothing here is evidence about a model.  It computes numbers from predictions; whether those predictions are evidence depends
on where they came from (see the evidence labels in `ml.validation.phase4a`).
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6                     # probability clip used only for logit / log loss
MIN_SUBGROUP_CLUSTERS = 8      # reporting rule, NOT a clinical or statistical sufficiency claim
MIN_SUBGROUP_PER_CLASS = 20    # rows of each class required before a subgroup number is shown
METRICS = ("roc_auc", "pr_auc", "brier", "log_loss", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "ppv", "npv",
           "calibration_slope", "calibration_intercept")
HIGHER_IS_BETTER = {"roc_auc": True, "pr_auc": True, "brier": False, "log_loss": False, "accuracy": True, "balanced_accuracy": True,
                    "sensitivity": True, "specificity": True, "ppv": True, "npv": True}


# ----------------------------------------------------------------------------- scalar metrics
def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def log_loss(y, p) -> float | None:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return None
    q = _clip(p)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def _logit(p) -> np.ndarray:
    q = _clip(p)
    return np.log(q / (1 - q))


def _fit_logistic(y: np.ndarray, x: np.ndarray | None, offset: np.ndarray) -> tuple[float, float | None] | None:
    """Unpenalised logistic fit  logit P(y=1) = a + b*x + offset  (b absent when x is None) by Newton-Raphson.
    Returns (a, b) or None when it does not converge or the data are separable."""
    n = len(y)
    X = np.ones((n, 1)) if x is None else np.column_stack([np.ones(n), x])
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta + offset
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        w = mu * (1 - mu)
        grad = X.T @ (y - mu)
        H = X.T @ (X * w[:, None]) + 1e-10 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    else:
        return None
    if np.any(np.abs(beta) > 30):          # quasi-complete separation: coefficients run away
        return None
    return float(beta[0]), (None if x is None else float(beta[1]))


def calibration_intercept_slope(y, p) -> dict:
    """Calibration slope b and intercept a from  logit P(y=1) = a + b*logit(p)  (slope), and the calibration-in-the-large
    intercept a0 from  logit P(y=1) = a0 + logit(p)  (slope fixed at 1).  A perfectly calibrated model has b = 1, a0 = 0.
    Either is None, with a reason, when it cannot be estimated.  Scores from class-weighted or otherwise re-balanced models are
    not probabilities, so a poor result there is expected and is reported as found."""
    y = np.asarray(y, dtype=float)
    out: dict = {"slope": None, "intercept_with_slope": None, "intercept": None, "reasons": {}}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out["reasons"]["all"] = "only one class present"
        return out
    lg = _logit(p)
    if float(np.std(lg)) < 1e-8:
        out["reasons"]["slope"] = "constant scores: slope is undefined"
    else:
        fit = _fit_logistic(y, lg, np.zeros(len(y)))
        if fit is None:
            out["reasons"]["slope"] = "no convergence or separable data"
        else:
            out["intercept_with_slope"], out["slope"] = fit
    fit0 = _fit_logistic(y, None, lg)
    if fit0 is None:
        out["reasons"]["intercept"] = "no convergence"
    else:
        out["intercept"] = fit0[0]
    return out


def confusion_at(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    return {"tp": int(((pred == 1) & (y == 1)).sum()), "fn": int(((pred == 0) & (y == 1)).sum()),
            "fp": int(((pred == 1) & (y == 0)).sum()), "tn": int(((pred == 0) & (y == 0)).sum())}


def _ratio(a: int, b: int) -> float | None:
    return None if b == 0 else a / b


def point_metrics(y, p, threshold: float) -> dict:
    """Every metric in METRICS at one fixed threshold (None where undefined)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    c = confusion_at(y, p, threshold)
    n = len(y)
    both = len(np.unique(y)) == 2
    sens, spec = _ratio(c["tp"], c["tp"] + c["fn"]), _ratio(c["tn"], c["tn"] + c["fp"])
    cal = calibration_intercept_slope(y, p) if both else {"slope": None, "intercept": None}
    return {"roc_auc": float(roc_auc_score(y, p)) if both else None, "pr_auc": float(average_precision_score(y, p)) if both else None,
            "brier": float(np.mean((p - y) ** 2)) if n else None, "log_loss": log_loss(y, p),
            "accuracy": _ratio(c["tp"] + c["tn"], n), "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": None if sens is None or spec is None else (sens + spec) / 2,
            "ppv": _ratio(c["tp"], c["tp"] + c["fp"]), "npv": _ratio(c["tn"], c["tn"] + c["fn"]),
            "calibration_slope": cal["slope"], "calibration_intercept": cal["intercept"]}


# ----------------------------------------------------------------------------- decision curve
def net_benefit(y, p, thresholds: Sequence[float]) -> list[dict]:
    """Decision-curve analysis (Vickers & Elkin).  net benefit of 'act if p >= t' =  TP/n - FP/n * t/(1-t); compared with
    'act on everyone' (prevalence - (1-prevalence) * t/(1-t)) and 'act on no one' (0).

    ASSUMPTION that no code can check: t is the threshold probability at which a clinician/patient is indifferent between
    acting and not acting.  Without a defined clinical action and a justified range of t, the curve is a diagnostic, not a
    clinical-utility result."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    rows = []
    prev = float(y.mean()) if n else None
    for t in thresholds:
        if not (0.0 < t < 1.0):
            raise ValueError("decision-curve thresholds must be in (0, 1)")
        odds = t / (1 - t)
        pred = p >= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        rows.append({"threshold": float(t), "net_benefit_model": (tp / n - fp / n * odds) if n else None,
                     "net_benefit_treat_all": (prev - (1 - prev) * odds) if n else None, "net_benefit_treat_none": 0.0,
                     "n_act": int(pred.sum())})
    return rows


# ----------------------------------------------------------------------------- cluster bootstrap for all metrics at once
def _cluster_index(clusters: np.ndarray) -> list[np.ndarray]:
    order = pd.Series(np.arange(len(clusters))).groupby(clusters, sort=True).apply(lambda s: s.to_numpy())
    return list(order.to_numpy())


def bootstrap_all(y, p_by_model: dict[str, np.ndarray], clusters, *, threshold: float, n_boot: int, seed: int,
                  alpha: float = 0.05) -> dict:
    """One set of cluster resamples used for every model and metric, so differences between models are PAIRED.
    Returns {model: {metric: array of resampled values (NaN where undefined)}} plus the cluster count."""
    y = np.asarray(y).astype(int)
    idx = _cluster_index(np.asarray(clusters))
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list]] = {m: {k: [] for k in METRICS} for m in p_by_model}
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), size=len(idx))
        rows = np.concatenate([idx[i] for i in pick])
        yy = y[rows]
        for m, pp in p_by_model.items():
            v = point_metrics(yy, np.asarray(pp)[rows], threshold)
            for k in METRICS:
                draws[m][k].append(np.nan if v[k] is None else v[k])
    return {"n_clusters": len(idx), "n_boot": n_boot, "alpha": alpha,
            "draws": {m: {k: np.asarray(a, dtype=float) for k, a in d.items()} for m, d in draws.items()}}


def _interval(a: np.ndarray, alpha: float, n_boot: int) -> dict:
    ok = a[~np.isnan(a)]
    out = {"n_valid": int(len(ok)), "n_discarded": int(n_boot - len(ok))}
    if len(ok) >= max(20, 0.5 * n_boot):
        out["lo"], out["hi"] = float(np.percentile(ok, 100 * alpha / 2)), float(np.percentile(ok, 100 * (1 - alpha / 2)))
    else:
        out["lo"] = out["hi"] = None
        out["note"] = "too many degenerate resamples for a reliable interval"
    return out


def comprehensive(y, p, clusters, *, threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """Point estimates + cluster-bootstrap intervals for every metric, confusion matrix with denominators, and the reliability
    table.  The threshold must have been fixed BEFORE these predictions were seen."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pt = point_metrics(y, p, threshold)
    bs = bootstrap_all(y, {"m": p}, clusters, threshold=threshold, n_boot=n_boot, seed=seed)
    d = bs["draws"]["m"]
    return {"n_rows": int(len(y)), "n_clusters": bs["n_clusters"], "threshold": threshold, "prevalence": float(y.mean()),
            "confusion": confusion_at(y, p, threshold),
            "metrics": {k: {"point": pt[k], **_interval(d[k], bs["alpha"], n_boot)} for k in METRICS},
            "calibration": calibration_intercept_slope(y, p),
            "intervals": {"method": "cluster bootstrap (resample subjects), percentile", "n_boot": n_boot, "seed": seed}}


def paired_differences(y, p_a, p_b, clusters, *, metrics: Sequence[str], threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """metric(a) - metric(b) for several metrics on the SAME resampled clusters (one bootstrap).  `a_better_with_interval_excluding_zero`
    accounts for direction (lower Brier / log loss is better).  If the interval contains 0 the data cannot separate the two."""
    for metric in metrics:
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}")
    y = np.asarray(y).astype(int)
    bs = bootstrap_all(y, {"a": np.asarray(p_a, float), "b": np.asarray(p_b, float)}, clusters, threshold=threshold, n_boot=n_boot, seed=seed)
    ma, mb = point_metrics(y, p_a, threshold), point_metrics(y, p_b, threshold)
    out = {}
    for metric in metrics:
        diff = bs["draws"]["a"][metric] - bs["draws"]["b"][metric]
        iv = _interval(diff, bs["alpha"], n_boot)
        point = None if ma[metric] is None or mb[metric] is None else ma[metric] - mb[metric]
        higher = HIGHER_IS_BETTER.get(metric)
        excludes_zero = iv["lo"] is not None and (iv["lo"] > 0 or iv["hi"] < 0)
        a_better = None
        if excludes_zero and higher is not None:
            a_better = (iv["lo"] > 0) if higher else (iv["hi"] < 0)
        out[metric] = {"metric": metric, "point_difference_a_minus_b": point, **iv, "interval_excludes_zero": excludes_zero,
                       "a_better_with_interval_excluding_zero": a_better, "n_clusters": bs["n_clusters"], "n_boot": n_boot}
    return out


def paired_difference(y, p_a, p_b, clusters, *, metric: str, threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    return paired_differences(y, p_a, p_b, clusters, metrics=[metric], threshold=threshold, n_boot=n_boot, seed=seed)[metric]


# ----------------------------------------------------------------------------- subgroups
def subgroup_report(df: pd.DataFrame, by: str, *, y="y", p="p", cluster="subject", threshold: float, n_boot: int = 500, seed: int = 0) -> list[dict]:
    """Per-level metrics.  A level is reported only when it has at least MIN_SUBGROUP_CLUSTERS clusters and
    MIN_SUBGROUP_PER_CLASS rows of each class; otherwise it is listed as insufficient.  These are reporting rules, not a claim
    that the displayed numbers are precise: intervals are shown and are usually wide."""
    out = []
    for level, g in df.groupby(by, sort=True, dropna=False):
        yy = g[y].astype(int)
        npos, nneg, nclu = int((yy == 1).sum()), int((yy == 0).sum()), int(g[cluster].nunique())
        base = {"subgroup": by, "level": None if pd.isna(level) else (level.item() if hasattr(level, "item") else level),
                "n_rows": int(len(g)), "n_clusters": nclu, "n_positive": npos, "n_negative": nneg}
        if nclu < MIN_SUBGROUP_CLUSTERS or npos < MIN_SUBGROUP_PER_CLASS or nneg < MIN_SUBGROUP_PER_CLASS:
            out.append({**base, "status": "insufficient",
                        "reason": f"needs >= {MIN_SUBGROUP_CLUSTERS} subjects and >= {MIN_SUBGROUP_PER_CLASS} rows of each class"})
        else:
            out.append({**base, "status": "reported",
                        "result": comprehensive(g[y].to_numpy(), g[p].to_numpy(), g[cluster].to_numpy(), threshold=threshold, n_boot=n_boot, seed=seed)})
    return out


# ----------------------------------------------------------------------------- recalibration (exploratory, cross-fitted)
def cross_fitted_recalibration(y, p, groups, *, k: int = 5, seed: int = 0) -> dict:
    """How much would a logistic (Platt-type) recalibration  logit P = a + b*logit(p)  help?  The recalibrator is fitted on the
    held-out scores of OTHER subjects and applied to each subject group's scores (group k-fold), so a subject is never
    recalibrated by a map that saw their own outcome.  EXPLORATORY: the scores being recalibrated are themselves out-of-fold
    predictions, and a map estimated this way is not a deployable calibration (that would need its own validation)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    g = np.asarray(groups)
    ug = np.unique(g)
    if len(ug) < k or len(np.unique(y)) < 2:
        return {"status": "not_estimable", "reason": "fewer subjects than folds or a single class"}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ug)
    fold_of = {s: i % k for i, s in enumerate(perm)}
    f = np.array([fold_of[s] for s in g])
    q = np.full(len(y), np.nan)
    lg = _logit(p)
    n_fail = 0
    for i in range(k):
        tr, te = f != i, f == i
        if len(np.unique(y[tr])) < 2 or float(np.std(lg[tr])) < 1e-8:
            n_fail += 1
            q[te] = p[te]
            continue
        fit = _fit_logistic(y[tr].astype(float), lg[tr], np.zeros(tr.sum()))
        if fit is None:
            n_fail += 1
            q[te] = p[te]
            continue
        a, b = fit
        q[te] = 1.0 / (1.0 + np.exp(-(a + b * lg[te])))
    return {"status": "estimated", "n_folds": k, "n_folds_without_a_fit": n_fail, "recalibrated": q.tolist(),
            "brier_before": float(np.mean((p - y) ** 2)), "brier_after": float(np.mean((q - y) ** 2)),
            "log_loss_before": log_loss(y, p), "log_loss_after": log_loss(y, q),
            "calibration_before": calibration_intercept_slope(y, p), "calibration_after": calibration_intercept_slope(y, q),
            "roc_auc_before": float(roc_auc_score(y, p)), "roc_auc_after": float(roc_auc_score(y, q)),
            "label": "EXPLORATORY_NOT_A_DEPLOYABLE_CALIBRATION"}
