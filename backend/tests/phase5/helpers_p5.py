"""Synthetic fixtures for Phase 5 workflow tests. NOT evidence about any model or data set."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[2]


def risk_columns() -> list[str]:
    meta = json.loads((BACKEND / "ml/models/parkinsons_risk_classifier_metadata.json").read_text())
    voice = [c[len("delta_"):] for c in meta["feature_columns_in_order"] if c.startswith("delta_")]
    return ["subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS", *voice]


def forged_risk_csv(path: Path, seed: int = 0) -> Path:
    """Random numbers satisfying every structural check of the published risk file: 5,875 rows, 42 subjects, 22 columns, no NaN."""
    rng = np.random.default_rng(seed)
    cols = risk_columns()
    voice = cols[6:]
    sizes = [140] * 37 + [139] * 5
    rows = []
    for s, n in enumerate(sizes, start=1):
        base, slope = rng.uniform(15, 40), rng.normal(0.03, 0.02)
        age, sex = int(rng.integers(50, 80)), int(rng.integers(0, 2))
        for t in np.sort(rng.uniform(0, 180, n)):
            rows.append({"subject#": s, "age": age, "sex": sex, "test_time": float(t), "motor_UPDRS": 0.0,
                         "total_UPDRS": base + slope * t + rng.normal(0, 0.8), **{c: float(rng.normal(0, 1)) for c in voice}})
    df = pd.DataFrame(rows)[cols]
    assert len(df) == 5875 and df["subject#"].nunique() == 42 and df.shape[1] == 22
    df.to_csv(path, index=False)
    return path
