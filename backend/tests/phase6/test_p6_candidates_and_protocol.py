"""Phase 6 prespecification: the extension families are the ones written down, contain no data-dependent step outside the Pipeline,
fit without warnings, and the frozen records cannot drift.  Synthetic data; nothing here is evidence."""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.validation import candidates_p6 as C6
from ml.validation import phase4a as P
from ml.validation import phase6 as PH6

REG = Path(__file__).resolve().parents[2] / "ml/validation/protocol_registry"
P4A_SHA = "0b2f10a7ec885419d9a2e72ad80b9d4c60b76987ea71e654edcb157b86723e42"
P6_SHA = "7fd315bb10c38478c816c2bd552e8d799e0e36915ce027a50f49ab75b0606ec8"      # pinned the day p6-1 was frozen, before any real result


def test_p4a_protocol_is_untouched_by_phase6():
    assert P.PROTOCOL_SHA256 == P4A_SHA
    assert json.loads((REG / "p4a-1.json").read_text())["sha256"] == P4A_SHA
    assert PH6.PROTOCOL_P6["parent_protocol"]["sha256"] == P4A_SHA and PH6.PROTOCOL_P6["parent_protocol"]["relationship"].startswith("ADDITIVE")


def test_p6_protocol_hash_is_pinned_and_registry_copy_equals_it():
    assert PH6.PROTOCOL_P6_SHA256 == P6_SHA
    rec = json.loads((REG / "p6-1.json").read_text())
    assert rec["status"] == "FROZEN" and rec["frozen_before_any_real_result"] is True and rec["sha256"] == P6_SHA
    assert rec["protocol"] == PH6.PROTOCOL_P6
    assert hashlib.sha256(json.dumps(rec["protocol"], sort_keys=True).encode()).hexdigest() == P6_SHA


def test_p6_record_sets_no_clinical_target_or_threshold():
    pr = PH6.PROTOCOL_P6
    assert pr["acceptance_criteria"].startswith("REQUIRES_CLINICAL_INPUT")
    assert pr["decision_curve"]["interpretation"].startswith("REQUIRES_CLINICAL_INPUT")
    assert pr["threshold"] == P.PROTOCOL["threshold"] == 0.5


def test_extension_sets_are_a_subset_of_the_registered_p4a_sets_and_use_the_registered_columns():
    cols = [f"d{i}" for i in range(16)]
    sets = PH6.risk_feature_sets(cols)
    assert set(sets) == set(PH6.RISK_SETS) <= set(P.PROTOCOL["risk"]["feature_sets"])
    assert sets["TIME_BASELINE"] == ["test_time", "baseline_total_UPDRS"] and sets["VOICE_ONLY"] == cols


def _fit_all(cands, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # a deprecation or convergence warning is a failure, not noise
        for name, fac in cands.items():
            m = fac()
            assert isinstance(m, Pipeline), name
            m.fit(X, y)
            p = m.predict_proba(X)[:, 1]
            assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1, name


def test_risk_extension_families_fit_without_warnings_and_only_inside_pipelines():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"test_time": rng.uniform(0, 180, 400), "baseline_total_UPDRS": rng.uniform(10, 40, 400), "d": rng.normal(size=400)})
    y = (X.test_time / 180 + rng.normal(0, .4, 400) > .6).astype(int)
    cands = C6.risk_extension_candidates(list(X.columns))
    assert "HGB_monotone_time" in cands
    _fit_all(cands, X, y)
    first_steps = {n: type(f().steps[0][1]).__name__ for n, f in cands.items()}
    assert all(s in ("StandardScaler", "HistGradientBoostingClassifier") for s in first_steps.values()), first_steps


def test_monotone_family_exists_only_when_test_time_is_a_feature():
    assert "HGB_monotone_time" not in C6.risk_extension_candidates(["d1", "d2"])
    assert "HGB_monotone_time" in C6.risk_extension_candidates(["test_time"])


def test_monotone_constraint_is_actually_monotone_in_test_time():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"test_time": rng.uniform(0, 180, 600), "x": rng.normal(size=600)})
    y = (rng.uniform(size=600) < 0.5).astype(int)                     # pure noise: an unconstrained tree would wiggle
    m = C6.risk_extension_candidates(list(X.columns))["HGB_monotone_time"]().fit(X, y)
    grid = pd.DataFrame({"test_time": np.linspace(0, 180, 60), "x": 0.0})
    assert (np.diff(m.predict_proba(grid)[:, 1]) >= -1e-12).all()


def test_disease_extension_families_fit_without_warnings():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(120, 22)))
    _fit_all(C6.disease_extension_candidates(), X, rng.integers(0, 2, 120))


def test_every_family_has_a_written_hypothesis():
    names = list(C6.risk_extension_candidates(["test_time"])) + list(C6.disease_extension_candidates())
    prefixes = {n.split("_C")[0] if "_C" in n else n for n in names}
    for pre in prefixes:
        assert any(h.startswith(pre) for h in C6.HYPOTHESES), pre


def test_fast_sets_are_subsets():
    assert set(C6.risk_extension_candidates(["test_time"], fast=True)) <= set(C6.risk_extension_candidates(["test_time"]))
    assert set(C6.disease_extension_candidates(fast=True)) <= set(C6.disease_extension_candidates())
