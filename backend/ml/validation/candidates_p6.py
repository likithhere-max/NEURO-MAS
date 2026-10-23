"""
Phase 6 extension candidates (protocol p6-1-ext).  Chosen and written down BEFORE any real-data result existed, from general
modelling considerations only (no access to held-out evaluation results).  Same rules as `candidates.py`: every data-dependent
step is inside the Pipeline, hyperparameter choices are separate named candidates so the nested inner loop picks among them
from training rows only.

A kernel SVM was considered and left out: its probability output in scikit-learn 1.9 is deprecated, and the replacement
(`CalibratedClassifierCV`) would calibrate on ungrouped rows of the training subjects.

Each family has a stated hypothesis.  A family that does not beat its comparator on paired held-out predictions is reported as
not better; it is not hidden and not replaced.
"""
from __future__ import annotations

from typing import Sequence

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

HYPOTHESES = {
    "LR_unweighted_C*": "Without class re-weighting, scores are closer to probabilities; calibration (Brier, log loss, slope) improves with similar ranking.",
    "LR_enet_C*": "Elastic-net shrinkage helps when many correlated voice deltas are present (risk FULL / VOICE_ONLY; disease 22 features).",
    "LR_spline_C*": "The within-patient trend in time / baseline severity is non-linear; a spline basis captures it better than a straight line (risk only).",
    "HGB_regularised": "A heavily regularised tree ensemble can capture interactions without the variance of the unregularised small HGB.",
    "HGB_monotone_time": "Worsening should not become less likely as time since recruitment grows; a monotone constraint reduces variance (risk sets containing test_time only).",
}


def _lr(C, *, balanced: bool, penalty: str = "l2"):
    kw = dict(C=C, max_iter=5000, random_state=0, class_weight="balanced" if balanced else None)
    if penalty == "elasticnet":
        kw.update(solver="saga", l1_ratio=0.5)          # scikit-learn >= 1.8: an l1_ratio in (0, 1) means elastic net; `penalty` is deprecated
    return lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(**kw))])


def _spline_lr(C):
    return lambda: Pipeline([("s", StandardScaler()), ("b", SplineTransformer(n_knots=4, degree=3, extrapolation="linear")),
                             ("c", LogisticRegression(C=C, max_iter=5000, random_state=0))])


def _hgb(**kw):
    return lambda: Pipeline([("c", HistGradientBoostingClassifier(random_state=0, **kw))])


def risk_extension_candidates(columns: Sequence[str], fast: bool = False) -> dict:
    """`columns` are the feature names of the set being fitted (needed only to attach a monotone constraint to test_time)."""
    c = {f"LR_unweighted_C{C}": _lr(C, balanced=False) for C in (0.01, 0.1, 1.0)}
    c.update({f"LR_enet_C{C}": _lr(C, balanced=True, penalty="elasticnet") for C in (0.01, 0.1)})
    c.update({f"LR_spline_C{C}": _spline_lr(C) for C in (0.01, 0.1)})
    c["HGB_regularised"] = _hgb(max_depth=3, max_iter=60, learning_rate=0.05, min_samples_leaf=200, l2_regularization=5.0, class_weight="balanced")
    if "test_time" in columns:
        c["HGB_monotone_time"] = _hgb(max_depth=3, max_iter=60, learning_rate=0.05, min_samples_leaf=200, l2_regularization=5.0,
                                      class_weight="balanced", monotonic_cst={"test_time": 1})
    if fast:
        keep = ("LR_unweighted_C0.1", "LR_enet_C0.1", "HGB_regularised")
        c = {k: v for k, v in c.items() if k in keep}
    return c


def disease_extension_candidates(fast: bool = False) -> dict:
    c = {f"LR_unweighted_C{C}": _lr(C, balanced=False) for C in (0.1, 1.0)}
    c.update({f"LR_enet_C{C}": _lr(C, balanced=True, penalty="elasticnet") for C in (0.1, 1.0)})
    if fast:
        c = {k: v for k, v in c.items() if k in ("LR_unweighted_C1.0", "LR_enet_C0.1")}
    return c
