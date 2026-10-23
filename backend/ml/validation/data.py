"""
Dataset loading + structural verification for the two UCI Parkinson's datasets (Phase 2, B3/B4).

Nothing is downloaded implicitly during tests. `fetch()` is only called by the CLI when the user runs it.
Every loader VERIFIES the file against the dataset's published structure and raises DatasetVerificationError
(status "failed" in the results JSON) instead of silently analysing a corrupted / wrong / modified file.

Provenance (see docs/phase2/DATASET_REGISTER.md):
  disease : UCI "Parkinsons" (id 174), Little et al. 2007. 195 recordings, 22 voice features + status + name.
  risk    : UCI "Parkinsons Telemonitoring" (id 189), Tsanas et al. 2009/2010. 5875 recordings, 42 patients.
The expected counts below are those documented by the dataset publisher; they are used as integrity checks,
NOT as targets to tune against.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data" / "cache"
LOCK = HERE.parent / "data" / "DATASETS.lock.json"

SOURCES = {
    "parkinsons": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
    "parkinsons_updrs": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
}

DISEASE_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
    "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]
EXPECTED = {
    "parkinsons": {"rows": 195, "pd_rows": 147, "healthy_rows": 48, "n_columns": 24},
    "parkinsons_updrs": {"rows": 5875, "subjects": 42, "n_columns": 22},
}


class DatasetVerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(name: str, *, timeout: int = 60) -> Path:
    """Download one dataset into ml/data/cache and record its SHA-256 in DATASETS.lock.json (trust on first use:
    later runs refuse a file whose hash differs from the recorded one)."""
    url = SOURCES[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / Path(url).name
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed https URL list above
            dest.write_bytes(r.read())
    digest = sha256_file(dest)
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    if name in lock and lock[name]["sha256"] != digest:
        raise DatasetVerificationError(f"{name}: SHA-256 {digest} differs from the locked value {lock[name]['sha256']}")
    lock.setdefault(name, {"url": url, "sha256": digest, "bytes": dest.stat().st_size})
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return dest


def check_lock(name: str, path: Path) -> str:
    digest = sha256_file(path)
    if LOCK.exists():
        lock = json.loads(LOCK.read_text())
        if name in lock and lock[name]["sha256"] != digest:
            raise DatasetVerificationError(f"{name}: file hash {digest} differs from DATASETS.lock.json")
    return digest


# ----------------------------------------------------------------------------- disease dataset
def verify_disease_frame(df: pd.DataFrame, *, strict_counts: bool = True) -> dict:
    exp = EXPECTED["parkinsons"]
    missing = [c for c in ["name", "status", *DISEASE_FEATURES] if c not in df.columns]
    if missing:
        raise DatasetVerificationError(f"disease: missing columns {missing}")
    if df[["status", *DISEASE_FEATURES]].isna().any().any():
        raise DatasetVerificationError("disease: missing values present (not expected in the published file)")
    if not set(df["status"].unique()) <= {0, 1}:
        raise DatasetVerificationError("disease: status must be 0/1")
    bad_names = [n for n in df["name"] if not re.fullmatch(r"phon_R\d+_S\d+_\d+", str(n))]
    if bad_names:
        raise DatasetVerificationError(f"disease: unparseable recording names, e.g. {bad_names[:3]}")
    if strict_counts and (len(df) != exp["rows"] or int(df["status"].sum()) != exp["pd_rows"]):
        raise DatasetVerificationError(
            f"disease: expected {exp['rows']} rows / {exp['pd_rows']} PD rows, got {len(df)} / {int(df['status'].sum())}")
    if not np.isfinite(df[DISEASE_FEATURES].to_numpy(float)).all():
        raise DatasetVerificationError("disease: non-finite feature values")
    out = df.copy()
    out["subject"] = out["name"].str.extract(r"(S\d+)_\d+$")[0]
    per = out.groupby("subject")["status"].agg(["nunique", "first", "size"])
    mixed = per.index[per["nunique"] > 1].tolist()
    if mixed:
        raise DatasetVerificationError(f"disease: subjects with mixed labels (ID parsing wrong?): {mixed}")
    dup = int(out.duplicated(subset=DISEASE_FEATURES).sum())
    return {"n_rows": len(out), "n_subjects": int(per.shape[0]), "subjects_pd": int((per["first"] == 1).sum()),
            "subjects_healthy": int((per["first"] == 0).sum()), "row_prevalence_pd": float(out["status"].mean()),
            "recordings_per_subject": [int(per["size"].min()), int(per["size"].max())],
            "duplicate_feature_rows": dup, "missing_values": 0}


def load_disease(path: Path, *, strict_counts: bool = True):
    path = Path(path)
    digest = check_lock("parkinsons", path)
    df = pd.read_csv(path)
    info = verify_disease_frame(df, strict_counts=strict_counts)
    df["subject"] = df["name"].str.extract(r"(S\d+)_\d+$")[0]
    info["sha256"] = digest
    return df, info


# ----------------------------------------------------------------------------- risk dataset
RISK_NON_VOICE = ("subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS")
WORSENING_THRESHOLD = 3.0  # PROVISIONAL engineering parameter (not a validated MCID); configurable


def verify_risk_frame(df: pd.DataFrame, *, strict_counts: bool = True) -> dict:
    exp = EXPECTED["parkinsons_updrs"]
    df = df.rename(columns=lambda c: c.strip())
    need = ["subject#", "test_time", "total_UPDRS"]
    if any(c not in df.columns for c in need):
        raise DatasetVerificationError(f"risk: missing columns {[c for c in need if c not in df.columns]}")
    if df.isna().any().any():
        raise DatasetVerificationError("risk: missing values present")
    if strict_counts and (len(df) != exp["rows"] or df["subject#"].nunique() != exp["subjects"]):
        raise DatasetVerificationError(
            f"risk: expected {exp['rows']} rows / {exp['subjects']} subjects, got {len(df)} / {df['subject#'].nunique()}")
    return {"n_rows_raw": len(df), "n_subjects": int(df["subject#"].nunique()),
            "visits_per_subject": [int(df.groupby('subject#').size().min()), int(df.groupby('subject#').size().max())],
            "test_time_days": [float(df["test_time"].min()), float(df["test_time"].max())],
            "missing_values": 0}


def build_risk_frame(df: pd.DataFrame, *, threshold: float = WORSENING_THRESHOLD):
    """Replicates the training definition: label = total_UPDRS(t) - total_UPDRS(first visit) >= threshold.
    Features are voice DELTAS from the first visit plus test_time and baseline_total_UPDRS.

    Leakage notes (documented, not hidden):
      * The label and the delta features are measured at the SAME visit t -> this is a CONCURRENT
        classification ('is this visit worse than baseline?'), not a forecast of future worsening.
      * baseline_total_UPDRS is a component of the label definition; it is legitimately known at inference
        time (the patient's own first visit) but it makes the target partly predictable from baseline severity
        and elapsed time with NO voice information - hence the mandatory ablations."""
    df = df.rename(columns=lambda c: c.strip()).sort_values(["subject#", "test_time"]).reset_index(drop=True)
    voice = [c for c in df.columns if c not in RISK_NON_VOICE]
    first = df.groupby("subject#", as_index=False).first().set_index("subject#")
    df["baseline_total_UPDRS"] = df["subject#"].map(first["total_UPDRS"])
    for c in voice:
        df[f"delta_{c}"] = df[c] - df["subject#"].map(first[c])
    is_first = df["test_time"] == df.groupby("subject#")["test_time"].transform("min")
    df = df[~is_first].reset_index(drop=True)
    df["worsening"] = ((df["total_UPDRS"] - df["baseline_total_UPDRS"]) >= threshold).astype(int)
    return df, [f"delta_{c}" for c in voice]


def load_risk(path: Path, *, strict_counts: bool = True, threshold: float = WORSENING_THRESHOLD):
    path = Path(path)
    digest = check_lock("parkinsons_updrs", path)
    raw = pd.read_csv(path)
    info = verify_risk_frame(raw, strict_counts=strict_counts)
    df, delta_cols = build_risk_frame(raw, threshold=threshold)
    info.update({"n_rows": len(df), "positive_rate": float(df["worsening"].mean()), "sha256": digest,
                 "worsening_threshold_points": threshold,
                 "subjects_with_both_classes": int((df.groupby("subject#")["worsening"].nunique() > 1).sum())})
    return df, delta_cols, info
