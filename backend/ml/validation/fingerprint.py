"""
Dataset authentication against the shipped model artifacts (Phase 4A).

Problem. The Phase 2 runner labelled ANY csv passed with `--csv` as "REAL_DATA_RUN" once it had the right shape
(195 rows / 147 PD rows, or 5,875 rows / 42 subjects). A synthetic or altered file of that shape would therefore have been
reported as real data. A SHA-256 lock only protects against change after first use ("trust on first use"), and
`ml/data/DATASETS.lock.json` has never been populated because the download has never succeeded.

Idea. Each shipped artifact contains a `StandardScaler` that was FITTED ON THE TRAINING MATRIX. Its `n_samples_seen_`,
`mean_` and `scale_` are therefore a fingerprint of that matrix: a different file (or the same file processed
differently) essentially cannot reproduce 18 or 22 means and standard deviations to ~1e-6 of a standard deviation.
`authenticate()` recomputes the same statistics from a candidate file, using the project's own feature construction, and
compares them.

What a MATCH means:   the file reproduces the matrix the shipped model was trained on (the same rows, in the sense of the
                      first two moments of every feature, and the same preprocessing).
What it does NOT mean: that the data are clinically meaningful, that labels are right (labels are not part of the
                      fingerprint), or that a model evaluated on them is clinically valid.
A MISMATCH means:     the file is not the training matrix. It can still be a legitimate different data set (for example
                      an external cohort), but results on it must not be presented as a reproduction of the reported figures.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.validation import data as D

MODELS = Path(__file__).resolve().parent.parent / "models"
STEMS = {"disease": "parkinsons_demo_classifier", "risk": "parkinsons_risk_classifier"}

AUTHENTICATED = "AUTHENTICATED_AS_TRAINING_DATA"
NOT_TRAINING_DATA = "NOT_THE_TRAINING_DATA"
NOT_COMPARABLE = "NOT_COMPARABLE"

EVIDENCE_AUTHENTICATED = "REAL_DATA_AUTHENTICATED_AGAINST_SHIPPED_ARTIFACT"
EVIDENCE_UNAUTHENTICATED = "UNAUTHENTICATED_USER_FILE_NOT_A_REPRODUCTION_OF_THE_REPORTED_FIGURES"

# deviations are measured in units of the artifact's own standard deviation, so the tolerance is scale-free
MEAN_TOL_SD = 1e-6
SCALE_REL_TOL = 1e-6


class FingerprintError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_artifact(kind: str, models_dir: Path = MODELS):
    """Load a shipped pipeline ONLY if its SHA-256 matches MANIFEST.json, via the restricted unpickler."""
    from app.ml.safe_load import restricted_joblib_load
    stem = STEMS[kind]
    jl = Path(models_dir) / f"{stem}.joblib"
    man = json.loads((Path(models_dir) / "MANIFEST.json").read_text()).get("sha256", {})
    if man.get(jl.name) != _sha256(jl):
        raise FingerprintError(f"{jl.name}: SHA-256 does not match MANIFEST.json; refusing to load")
    meta = json.loads((Path(models_dir) / f"{stem}_metadata.json").read_text())
    return restricted_joblib_load(jl), meta


def find_scaler(model):
    """The pipeline's StandardScaler, found BY TYPE (retrained pipelines name their steps 's'/'c'/'k', the legacy ones 'scaler'); None if the
    pipeline has no scaler (e.g. a gradient-boosting winner)."""
    from sklearn.preprocessing import StandardScaler
    steps = getattr(model, "named_steps", None) or {}
    named = steps.get("scaler")
    if isinstance(named, StandardScaler):
        return named
    for step in steps.values():
        if isinstance(step, StandardScaler):
            return step
    return None


def artifact_fingerprint(kind: str, models_dir: Path = MODELS) -> dict:
    """Training-matrix moments of the shipped artifact: from its StandardScaler if it has one, otherwise from the provenance sidecar
    (`<stem>_provenance.json`, `preprocessing`), which new artifacts compute directly from the training matrix."""
    model, meta = load_artifact(kind, models_dir)
    feats = list(meta["feature_columns_in_order"])
    sc = find_scaler(model)
    if sc is not None:
        n, mean, scale = int(sc.n_samples_seen_), [float(v) for v in sc.mean_], [float(v) for v in sc.scale_]
    else:
        side = Path(models_dir) / f"{STEMS[kind]}_provenance.json"
        if not side.exists():
            raise FingerprintError("the pipeline has no StandardScaler and there is no provenance sidecar: the training matrix cannot be fingerprinted")
        pre = json.loads(side.read_text())["preprocessing"]
        n, mean, scale = int(pre["n_samples_seen"]), [float(v) for v in pre["mean"]], [float(v) for v in pre["scale"]]
    if len(feats) != len(mean) or len(feats) != len(scale):
        raise FingerprintError("metadata feature list and fingerprint dimension disagree")
    return {"n": n, "features": feats, "mean": mean, "scale": scale}


def matrix_for(kind: str, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rebuild the training matrix from a RAW file with the project's own feature construction."""
    if kind == "disease":
        return df[features]
    if kind == "risk":
        built, _ = D.build_risk_frame(df)
        return built[features]
    raise ValueError(kind)


def data_fingerprint(kind: str, raw: pd.DataFrame, features: list[str]) -> dict:
    X = matrix_for(kind, raw, features).to_numpy(float)
    return {"n": int(len(X)), "features": list(features), "mean": X.mean(axis=0).tolist(),
            "scale": X.std(axis=0, ddof=0).tolist()}   # StandardScaler uses the population SD


def compare(expected: dict, observed: dict, *, mean_tol_sd: float = MEAN_TOL_SD, scale_rel_tol: float = SCALE_REL_TOL) -> dict:
    if expected["features"] != observed["features"]:
        return {"status": NOT_COMPARABLE, "reason": "feature lists differ"}
    mean_dev = np.abs(np.array(observed["mean"]) - np.array(expected["mean"])) / np.array(expected["scale"])
    scale_dev = np.abs(np.array(observed["scale"]) / np.array(expected["scale"]) - 1.0)
    ok = (observed["n"] == expected["n"] and float(mean_dev.max()) <= mean_tol_sd and float(scale_dev.max()) <= scale_rel_tol)
    worst = int(np.argmax(np.maximum(mean_dev / mean_tol_sd, scale_dev / scale_rel_tol)))
    return {"status": AUTHENTICATED if ok else NOT_TRAINING_DATA,
            "n_expected": expected["n"], "n_observed": observed["n"],
            "max_mean_deviation_in_sd_units": float(mean_dev.max()), "max_scale_relative_deviation": float(scale_dev.max()),
            "tolerances": {"mean_sd_units": mean_tol_sd, "scale_relative": scale_rel_tol},
            "worst_feature": expected["features"][worst], "n_features_compared": len(expected["features"])}


def authenticate(kind: str, csv_path: Path, models_dir: Path = MODELS) -> dict:
    """Never raises for a bad file: returns status NOT_COMPARABLE with the reason, so callers fail closed to 'unauthenticated'."""
    try:
        exp = artifact_fingerprint(kind, models_dir)
        raw = pd.read_csv(csv_path)
        raw = raw.rename(columns=lambda c: c.strip())
        obs = data_fingerprint(kind, raw, exp["features"])
        out = compare(exp, obs)
    except Exception as e:  # noqa: BLE001 - authentication must fail closed, whatever the reason
        out = {"status": NOT_COMPARABLE, "reason": f"{type(e).__name__}: {e}"}
    out["file_sha256"] = _sha256(csv_path) if Path(csv_path).exists() else None
    out["kind"] = kind
    return out


def evidence_status(auth: dict) -> str:
    return EVIDENCE_AUTHENTICATED if auth.get("status") == AUTHENTICATED else EVIDENCE_UNAUTHENTICATED
