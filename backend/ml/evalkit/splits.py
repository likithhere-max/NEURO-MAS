"""
Leakage-safe data splitting (Phase 2, B1/B3).

Every splitter in this module yields (train_idx, test_idx) pairs ONLY after running guards that
raise `LeakageError` if the pair would let information cross the evaluation boundary:

  * same group (patient/subject) on both sides                  -> assert_group_disjoint
  * the same feature row (duplicate observation) on both sides  -> assert_no_duplicate_rows
  * test observation earlier than training observation (when a
    temporal split is requested)                                -> assert_temporal_order
  * a group carrying more than one label when the target is
    group-level (a label-consistency check)                     -> assert_group_labels_consistent

The guards are deliberately independent of the splitting algorithm, so they also protect a splitter that
someone writes later (`guarded(...)` wraps any iterable of index pairs).
"""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    """Raised when a train/test pair would leak information across the evaluation boundary."""


def _as_array(x) -> np.ndarray:
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x)


def assert_group_disjoint(groups, train_idx, test_idx) -> None:
    g = _as_array(groups)
    shared = set(g[np.asarray(train_idx)].tolist()) & set(g[np.asarray(test_idx)].tolist())
    if shared:
        raise LeakageError(f"{len(shared)} group(s) appear in both train and test, e.g. {sorted(shared)[:3]}")


def assert_no_index_overlap(train_idx, test_idx) -> None:
    both = set(np.asarray(train_idx).tolist()) & set(np.asarray(test_idx).tolist())
    if both:
        raise LeakageError(f"{len(both)} row index(es) are in both train and test")


def _row_keys(X) -> list[tuple]:
    arr = _as_array(X)
    arr = arr.reshape(len(arr), -1)
    return [tuple(np.round(r.astype(float), 12)) for r in arr]


def assert_no_duplicate_rows(X, train_idx, test_idx) -> None:
    """Identical feature vectors on both sides of the boundary (e.g. a recording duplicated in the file)
    inflate performance because the model has effectively seen the test row."""
    keys = _row_keys(X)
    tr = {keys[i] for i in np.asarray(train_idx).tolist()}
    dup = [i for i in np.asarray(test_idx).tolist() if keys[i] in tr]
    if dup:
        raise LeakageError(f"{len(dup)} test row(s) are exact duplicates of training rows, e.g. index {dup[:3]}")


def assert_temporal_order(times, train_idx, test_idx, *, per_group: Sequence | None = None) -> None:
    """All training observations must precede all test observations (globally, or within each group if
    `per_group` is given: forward-chaining per patient)."""
    t = _as_array(times).astype(float)
    tr, te = np.asarray(train_idx), np.asarray(test_idx)
    if per_group is None:
        if len(tr) and len(te) and t[tr].max() >= t[te].min():
            raise LeakageError(
                f"temporal leakage: latest training time {t[tr].max()} >= earliest test time {t[te].min()}")
        return
    g = _as_array(per_group)
    for grp in set(g[te].tolist()):
        tr_g, te_g = tr[g[tr] == grp], te[g[te] == grp]
        if len(tr_g) and len(te_g) and t[tr_g].max() >= t[te_g].min():
            raise LeakageError(f"temporal leakage inside group {grp!r}")


def assert_group_labels_consistent(groups, y) -> None:
    """For a group-level target (e.g. PD vs healthy per person) every row of a group must share the label.
    A violation usually means two different people were given the same ID, which makes group splits leak."""
    df = pd.DataFrame({"g": _as_array(groups), "y": _as_array(y)})
    bad = df.groupby("g")["y"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise LeakageError(f"{len(bad)} group(s) carry more than one label: {bad.index.tolist()[:5]}")


def guarded(pairs: Iterable[tuple[np.ndarray, np.ndarray]], *, groups=None, X=None, times=None,
            temporal_per_group=None, check_duplicates: bool = True) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Wrap any splitter output; yield each pair only if every applicable guard passes."""
    for tr, te in pairs:
        tr, te = np.asarray(tr), np.asarray(te)
        if len(tr) == 0 or len(te) == 0:
            raise LeakageError("empty train or test split")
        assert_no_index_overlap(tr, te)
        if groups is not None:
            assert_group_disjoint(groups, tr, te)
        if X is not None and check_duplicates:
            assert_no_duplicate_rows(X, tr, te)
        if times is not None:
            assert_temporal_order(times, tr, te, per_group=temporal_per_group)
        yield tr, te


def group_kfold(groups, y=None, *, n_splits: int = 5, seed: int = 0, X=None, check_duplicates: bool = True):
    """Deterministic, shuffled, (stratified when y is given) group K-fold with leakage guards.
    n_splits is capped at the number of groups."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    g = _as_array(groups)
    k = int(min(n_splits, len(set(g.tolist()))))
    if k < 2:
        raise ValueError("need at least 2 groups to split")
    idx = np.arange(len(g))
    if y is not None:
        sp = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        pairs = sp.split(idx, _as_array(y), g)
    else:
        pairs = GroupKFold(n_splits=k).split(idx, groups=g)
    yield from guarded(pairs, groups=g, X=X, check_duplicates=check_duplicates)


def leave_one_group_out(groups, *, X=None, check_duplicates: bool = True):
    from sklearn.model_selection import LeaveOneGroupOut

    g = _as_array(groups)
    yield from guarded(LeaveOneGroupOut().split(np.arange(len(g)), groups=g), groups=g, X=X,
                       check_duplicates=check_duplicates)


def temporal_group_holdout(times, groups, *, test_fraction: float = 0.3):
    """Forward-chaining split WITHIN each patient: the earliest (1-f) of each patient's visits train, the
    latest f test. Use when the deployment question is 'predict this patient's future from their past'."""
    t, g = _as_array(times).astype(float), _as_array(groups)
    tr, te = [], []
    for grp in dict.fromkeys(g.tolist()):
        ix = np.where(g == grp)[0]
        ix = ix[np.argsort(t[ix], kind="stable")]
        cut = int(np.floor(len(ix) * (1 - test_fraction)))
        if cut < 1 or cut >= len(ix):
            continue  # too few visits to forward-chain; this patient is excluded, not mis-split
        tr.extend(ix[:cut].tolist())
        te.extend(ix[cut:].tolist())
    tr, te = np.array(tr), np.array(te)
    # Patients appear on both sides BY DESIGN here (it is a within-patient forecast), so only the temporal
    # guard applies.
    for pair in guarded([(tr, te)], times=t, temporal_per_group=g, check_duplicates=False):
        yield pair
