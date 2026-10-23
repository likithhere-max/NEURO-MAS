"""
Nested, group-aware cross-validation (Phase 2, B1).

Outer loop  : estimates the performance of the WHOLE procedure "select a model, fit it, predict".
Inner loop  : model (and threshold) selection, run ONLY on the outer-training rows.

Guarantees (enforced and unit-tested in tests/test_ml_evalkit.py):
  * outer test rows are never passed to any `fit`, selection, scaling, imputation, feature selection or
    threshold choice;
  * outer and inner splits are group-disjoint (patients never straddle a boundary);
  * candidate estimators are produced by factories, and the factory returns a Pipeline containing ALL data-
    dependent preprocessing (scaler, imputer, feature selector), so that these are re-fitted on every training
    split and never see held-out data.

What the outer estimate means: the performance of the selection PROCEDURE on new patients drawn like the
sample. It is not the performance of any particular saved model, and with ~30 subjects it is very uncertain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .splits import (LeakageError, assert_group_disjoint, assert_group_labels_consistent, group_kfold,
                     leave_one_group_out)


@dataclass
class FoldRecord:
    fold: int
    test_groups: list
    n_train: int
    n_test: int
    chosen: str
    inner_scores: dict
    selection_status: str
    threshold: float


@dataclass
class NestedResult:
    oof: np.ndarray
    folds: list[FoldRecord] = field(default_factory=list)
    selection_metric: str = "roc_auc"
    threshold_policy: str = "fixed:0.5"
    oof_threshold: np.ndarray | None = None  # the threshold of each row's OWN outer fold

    def chosen_counts(self) -> dict:
        return pd.Series([f.chosen for f in self.folds]).value_counts().to_dict()

    def to_dict(self) -> dict:
        return {"selection_metric": self.selection_metric, "threshold_policy": self.threshold_policy,
                "n_folds": len(self.folds), "chosen_counts": self.chosen_counts(),
                "folds": [f.__dict__ for f in self.folds]}


def _score(metric: str, y: np.ndarray, p: np.ndarray):
    if metric == "roc_auc":
        return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))
    if metric == "neg_brier":
        return float(-brier_score_loss(y, p))
    raise ValueError(f"unknown selection metric {metric!r}")


def _proba1(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 over the observed scores (ties -> lowest)."""
    if len(np.unique(y)) < 2:
        return 0.5
    cand = np.unique(p)
    best_t, best_j = 0.5, -np.inf
    for t in cand:
        pred = p >= t
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        j = sens + spec - 1
        if j > best_j + 1e-12:
            best_t, best_j = float(t), j
    return best_t


def inner_select(candidates: dict[str, Callable], X: pd.DataFrame, y: pd.Series, groups: pd.Series, *,
                 inner_k: int = 5, metric: str = "roc_auc", seed: int = 0):
    """Select among candidates using group K-fold on the GIVEN rows only.
    Returns (chosen_name, {name: score|None}, {name: inner OOF probabilities}, status)."""
    scores, oofs = {}, {}
    yv = y.to_numpy()
    for name, factory in candidates.items():
        oof = np.full(len(y), np.nan)
        try:
            for tr, te in group_kfold(groups, y, n_splits=inner_k, seed=seed, X=X, check_duplicates=False):
                if len(np.unique(yv[tr])) < 2:
                    raise ValueError("inner training split has one class")
                m = factory()
                m.fit(X.iloc[tr], y.iloc[tr])
                oof[te] = _proba1(m, X.iloc[te])
            ok = not np.isnan(oof).any()
        except ValueError:
            ok = False
        scores[name] = _score(metric, yv, oof) if ok else None
        oofs[name] = oof if ok else None
    valid = {k: v for k, v in scores.items() if v is not None}
    if not valid:
        first = next(iter(candidates))
        return first, scores, oofs, "fallback_no_valid_inner_score"
    chosen = max(valid, key=lambda k: (valid[k], -list(candidates).index(k)))  # ties -> earlier candidate
    return chosen, scores, oofs, "selected"


def nested_group_cv(X: pd.DataFrame, y, groups, candidates: dict[str, Callable], *,
                    outer: str | int = "logo", inner_k: int = 5, selection_metric: str = "roc_auc",
                    threshold_policy: str = "fixed:0.5", seed: int = 0,
                    group_level_target: bool = True, check_duplicates: bool = True) -> NestedResult:
    """`check_duplicates=False` is for LOW-DIMENSIONAL baseline feature sets (e.g. age+sex), where different subjects legitimately share an
    identical feature vector. Splits depend only on groups/labels/seed, so the duplicate guard run on the full feature matrix covers the
    same folds; callers that disable it must have run that full-matrix check."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a DataFrame (feature names are part of the contract)")
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    g = pd.Series(np.asarray(groups)).reset_index(drop=True)
    if not (len(X) == len(y) == len(g)):
        raise ValueError("X, y, groups must have equal length")
    if group_level_target:
        assert_group_labels_consistent(g, y)
    outer_pairs = list(leave_one_group_out(g, X=X, check_duplicates=check_duplicates) if outer == "logo"
                       else group_kfold(g, y, n_splits=int(outer), seed=seed, X=X, check_duplicates=check_duplicates))

    oof = np.full(len(y), np.nan)
    folds: list[FoldRecord] = []
    for k, (tr, te) in enumerate(outer_pairs):
        assert_group_disjoint(g, tr, te)  # belt and braces: the splitter already checks
        Xtr, ytr, gtr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True), g.iloc[tr].reset_index(drop=True)
        name, scores, oofs, status = inner_select(candidates, Xtr, ytr, gtr, inner_k=inner_k,
                                                  metric=selection_metric, seed=seed + k)
        if threshold_policy == "youden_inner" and oofs.get(name) is not None:
            thr = youden_threshold(ytr.to_numpy(), oofs[name])
        elif threshold_policy.startswith("fixed:"):
            thr = float(threshold_policy.split(":", 1)[1])
        elif threshold_policy == "youden_inner":
            thr = 0.5
        else:
            raise ValueError(f"unknown threshold policy {threshold_policy!r}")
        model = candidates[name]()
        model.fit(Xtr, ytr)
        oof[te] = _proba1(model, X.iloc[te])
        folds.append(FoldRecord(fold=k, test_groups=sorted(set(g.iloc[te].tolist()), key=str), n_train=len(tr),
                                n_test=len(te), chosen=name, inner_scores=scores, selection_status=status,
                                threshold=thr))
    if np.isnan(oof).any():
        raise LeakageError("some rows never appeared in an outer test fold")
    # per-row threshold actually applicable to each row's own outer fold (never a threshold learned elsewhere)
    thr_by_row = np.empty(len(y))
    for f, (_, te) in zip(folds, outer_pairs):
        thr_by_row[te] = f.threshold
    return NestedResult(oof=oof, folds=folds, selection_metric=selection_metric,
                        threshold_policy=threshold_policy, oof_threshold=thr_by_row)


def naive_selection_oof(candidates: dict[str, Callable], X: pd.DataFrame, y, groups) -> dict:
    """The BIASED protocol, kept only for comparison: every candidate is evaluated by LOSO and the best is
    reported. Its winner's score is optimistic because the winner was chosen on the same folds."""
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    g = pd.Series(np.asarray(groups)).reset_index(drop=True)
    out = {}
    for name, factory in candidates.items():
        oof = np.zeros(len(y))
        for tr, te in leave_one_group_out(g, X=X):
            m = factory()
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = _proba1(m, X.iloc[te])
        out[name] = oof
    return out
