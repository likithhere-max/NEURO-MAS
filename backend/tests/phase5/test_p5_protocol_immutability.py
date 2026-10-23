"""Phase 5 / Stage 3: the registered protocol is immutable, and its feature sets cannot leak identifiers or labels.
Synthetic data only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.validation import data as D
from ml.validation import phase4a as P
from tests.phase4.helpers_p4 import raw_risk_frame
from tests.phase5.helpers_p5 import risk_columns

REGISTRY = Path(P.__file__).resolve().parent / "protocol_registry"
FROZEN_SHA = "0b2f10a7ec885419d9a2e72ad80b9d4c60b76987ea71e654edcb157b86723e42"      # written down BEFORE any real result existed


def test_protocol_hash_is_pinned_to_the_value_registered_before_any_result():
    assert P.PROTOCOL_SHA256 == FROZEN_SHA


def test_protocol_equals_its_frozen_registry_copy():
    reg = json.loads((REGISTRY / "p4a-1.json").read_text())
    assert reg["status"] == "FROZEN" and reg["frozen_before_any_real_result"] is True and reg["sha256"] == FROZEN_SHA
    assert reg["protocol"] == json.loads(json.dumps(P.PROTOCOL))
    assert hashlib.sha256(json.dumps(reg["protocol"], sort_keys=True).encode()).hexdigest() == FROZEN_SHA


def test_every_registry_file_hash_matches_its_content():
    for p in REGISTRY.glob("*.json"):
        reg = json.loads(p.read_text())
        assert hashlib.sha256(json.dumps(reg["protocol"], sort_keys=True).encode()).hexdigest() == reg["sha256"], p.name


# ---------------------------------------------------------------------------------------------- no identifiers / labels as features
FORBIDDEN = {"subject#", "subject", "name", "status", "worsening", "total_UPDRS", "motor_UPDRS", "delta_target"}


def test_registered_risk_feature_sets_contain_no_identifier_label_or_outcome_column():
    sets = P.PROTOCOL["risk"]["feature_sets"]
    text = " ".join(sets.values())
    for bad in ("subject", "worsening", "motor_UPDRS"):
        assert bad not in text
    assert "total_UPDRS" not in text.replace("baseline_total_UPDRS", "")             # only the BASELINE (first-visit) score may appear
    meta = json.loads((Path(D.__file__).resolve().parents[1] / "models/parkinsons_risk_classifier_metadata.json").read_text())
    assert not (set(meta["feature_columns_in_order"]) & FORBIDDEN)
    assert not (set(D.DISEASE_FEATURES) & FORBIDDEN)


def test_risk_features_of_a_visit_depend_only_on_that_visit_and_the_patients_first_visit():
    """No information from other patients, or from the patient's later visits, enters a row's features."""
    raw = raw_risk_frame(seed=3, n_subjects=4, visits=12)
    voice = [c for c in raw.columns if c not in ("subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS")]
    base, dcols = D.build_risk_frame(raw)
    target = raw[raw["subject#"] == 2].sort_values("test_time").iloc[7]                # a later visit of patient 2
    changed = raw.copy()
    idx = changed.index[(changed["subject#"] == 2) & (changed["test_time"] == target["test_time"])][0]
    changed.loc[idx, voice[0]] += 5.0
    changed.loc[idx, "total_UPDRS"] += 9.0
    new, _ = D.build_risk_frame(changed)
    key = ["subject#", "test_time"]
    a, b = base.set_index(key), new.set_index(key)
    cols = dcols + ["baseline_total_UPDRS"]
    other = a.index.difference(pd.MultiIndex.from_tuples([(2, target["test_time"])], names=key))
    pd.testing.assert_frame_equal(a.loc[other, cols], b.loc[other, cols])              # every other row's features are untouched
    assert not np.allclose(a.loc[(2, target["test_time"]), dcols].to_numpy(float), b.loc[(2, target["test_time"]), dcols].to_numpy(float))


def test_the_label_uses_the_current_visit_and_the_first_visit_only():
    raw = raw_risk_frame(seed=4, n_subjects=3, visits=10)
    base, _ = D.build_risk_frame(raw)
    first = raw.sort_values(["subject#", "test_time"]).groupby("subject#").first()["total_UPDRS"]
    expect = ((base["total_UPDRS"] - base["subject#"].map(first)) >= D.WORSENING_THRESHOLD).astype(int)
    assert (base["worsening"].to_numpy() == expect.to_numpy()).all()


def test_risk_column_list_used_by_the_fixtures_is_the_published_22():
    assert len(risk_columns()) == 22 and risk_columns()[:3] == ["subject#", "age", "sex"]
