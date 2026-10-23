"""Shared fixtures for the Phase 4A tests.  EVERYTHING here is synthetic and is only ever used to exercise code paths and to
build adversarial cases for the evaluation harness. None of it is evidence about any real model or data set."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.validation import data as D
from ml.validation import fingerprint as F


def forged_disease_csv(path: Path, seed: int = 0) -> Path:
    """Random numbers that satisfy EVERY structural check of the published disease file:
    195 rows, 147 PD rows, 32 label-consistent subjects, parseable recording names, no NaN."""
    rng = np.random.default_rng(seed)
    rows = []
    sizes = [7] * 3 + [6] * 21 + [6] * 8                      # 24 PD subjects (147 rows) + 8 healthy (48 rows)
    labels = [1] * 24 + [0] * 8
    for s, (n, lab) in enumerate(zip(sizes, labels), start=1):
        for r in range(1, n + 1):
            rows.append({"name": f"phon_R01_S{s:02d}_{r}", "status": lab,
                         **{f: float(rng.normal(0, 1)) for f in D.DISEASE_FEATURES}})
    df = pd.DataFrame(rows)
    assert len(df) == 195 and int(df["status"].sum()) == 147
    df.to_csv(path, index=False)
    return path


def raw_risk_frame(seed: int = 1, n_subjects: int = 14, visits: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subjects + 1):
        base, slope = rng.uniform(15, 40), rng.normal(0.03, 0.02)
        for t in np.sort(rng.uniform(0, 180, visits)):
            rows.append({"subject#": s, "age": int(rng.integers(50, 80)), "sex": int(rng.integers(0, 2)), "test_time": float(t),
                         "motor_UPDRS": 0.0, "total_UPDRS": base + slope * t + rng.normal(0, 1.0),
                         "Jitter(%)": rng.normal(0.006, 0.001), "HNR": rng.normal(22, 2)})
    return pd.DataFrame(rows)


def make_artifact_dir(tmp: Path, kind: str, raw: pd.DataFrame) -> Path:
    """Fit a Pipeline([scaler, LR]) on the matrix derived from `raw` exactly as the project builds it and store it with a
    MANIFEST, mimicking ml/models/. Returns the directory."""
    tmp.mkdir(parents=True, exist_ok=True)
    stem = F.STEMS[kind]
    if kind == "disease":
        feats = list(D.DISEASE_FEATURES)
        X, y = raw[feats], raw["status"]
    else:
        built, dcols = D.build_risk_frame(raw)
        feats = dcols + ["test_time", "baseline_total_UPDRS"]
        X, y = built[feats], built["worsening"]
    pipe = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=500))]).fit(X, y)
    jl = tmp / f"{stem}.joblib"
    joblib.dump(pipe, jl)
    (tmp / f"{stem}_metadata.json").write_text(json.dumps({"feature_columns_in_order": feats}))
    (tmp / "MANIFEST.json").write_text(json.dumps({"sha256": {jl.name: F._sha256(jl)}}))
    return tmp
