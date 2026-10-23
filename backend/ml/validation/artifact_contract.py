"""
Model-artifact provenance contract (Phase 5, Stage 7 / "Phase H").

A provenance record is COMPLETE only if every required field carries a value.  A field that cannot be recovered is declared in
`unrecoverable` with a reason; that is honest, but the record is then INCOMPLETE_DECLARED, never COMPLETE.  A required field that is
simply absent makes the record INVALID.  Nothing is ever filled with a guess.

  build_training_provenance(...)   used by ml.training.train_final for newly trained artifacts (everything is known at training time)
  legacy_provenance(kind)          for the artifacts shipped before this contract existed (many fields are unrecoverable)
  validate_provenance(prov)        -> {"status": COMPLETE | INCOMPLETE_DECLARED | INVALID, "missing": [...], "unrecoverable": {...}}
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
from pathlib import Path

from app.ml import contract as K
from ml.validation import fingerprint as F

CONTRACT_VERSION = "1.1.0"      # 1.1.0 (Phase 6): + intended_use, split_identifiers, performance
MODELS = F.MODELS

REQUIRED_FIELDS = (
    "model_id", "model_version", "dataset_manifest", "protocol_version", "training_code_revision", "runtime_versions",
    "feature_schema", "feature_units", "preprocessing", "target_definition", "prediction_horizon", "training_date", "random_seeds",
    "validation_pointer", "calibration", "decision_threshold", "known_limitations", "artifact_checksum",
    "intended_use", "split_identifiers", "performance",
)
COMPLETE, INCOMPLETE_DECLARED, INVALID = "COMPLETE", "INCOMPLETE_DECLARED", "INVALID"


def _unit_from_name(feature: str) -> dict:
    """Units are taken ONLY from a unit written in the feature's own name. Everything else is declared unknown, not guessed."""
    base = feature[len("delta_"):] if feature.startswith("delta_") else feature
    for token, unit in (("(Hz)", "Hz"), ("(%)", "percent"), ("(dB)", "dB")):
        if token in base:
            return {"unit": unit, "status": "from_feature_name", "is_delta_vs_first_visit": feature.startswith("delta_")}
    if base == "test_time":
        return {"unit": K.TEST_TIME_UNIT, "status": "from_dataset_documentation_summarised", "convention": K.TEST_TIME_CONVENTION_ID}
    if base == "baseline_total_UPDRS":
        return {"unit": "UPDRS points", "status": "from_feature_name"}
    return {"unit": None, "status": "NOT_VERIFIED_no_unit_in_name_and_data_file_unavailable", "is_delta_vs_first_visit": feature.startswith("delta_")}


def feature_units(features: list[str]) -> dict:
    return {f: _unit_from_name(f) for f in features}


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd or Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def runtime_versions() -> dict:
    import numpy
    import pandas
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__}


def _task_text(kind: str) -> tuple[str, str]:
    return ((K.DISEASE_TARGET_DEFINITION, K.DISEASE_PREDICTION_HORIZON) if kind == "disease"
            else (K.RISK_TARGET_DEFINITION, K.RISK_PREDICTION_HORIZON))


def training_matrix_moments(X) -> dict:
    """Population (ddof=0) moments of the training matrix, computed directly from X so they exist whatever the pipeline contains."""
    import numpy as np
    a = np.asarray(X, float)
    return {"n_samples_seen": int(len(a)), "mean": a.mean(axis=0).tolist(), "scale": a.std(axis=0, ddof=0).tolist()}


INTENDED_USE_DOC = "docs/phase6/CLINICAL_INTENDED_USE.md"


def intended_use() -> dict:
    """Same for every artifact today: both models are research demonstrations.  A change of status needs the clinical decisions listed in
    CLINICAL_INTENDED_USE.md and is never made by editing this function alone."""
    return {"status": "RESEARCH_DEMONSTRATION", "clinical_use_permitted": False, "statement": K.DEMO_ONLY_LIMITATION, "document": INTENDED_USE_DOC}


def _sha_file(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validation_link(run_dir: Path | None, kind: str) -> tuple[dict | None, dict | None, dict]:
    """(split_identifiers, performance, unrecoverable) for an artifact whose registered validation run is in `run_dir`.
    Linked only if that run's experiment for `kind` is `completed`; otherwise the fields are declared unrecoverable with the reason.
    Metrics are not copied: the record points at the results file and pins it by SHA-256."""
    why = {}
    if run_dir is None:
        why = {"split_identifiers": "no validation run was supplied at training time", "performance": "no validation run was supplied at training time"}
        return None, None, why
    run_dir = Path(run_dir)
    mp = run_dir / "experiment_manifest.json"
    if not mp.exists():
        r = f"{run_dir} has no experiment_manifest.json"
        return None, None, {"split_identifiers": r, "performance": r}
    man = json.loads(mp.read_text())
    exp = man.get("experiments", {}).get(kind)
    if not exp or exp.get("status") != "completed":
        r = f"the {kind} validation experiment in {run_dir} is {exp.get('status') if exp else 'absent'}; nothing to link"
        return None, None, {"split_identifiers": r, "performance": r}
    folds = run_dir / f"{kind}_fold_assignments.json"
    res = next((run_dir / n for n in (f"{kind}_p4a.json", f"{kind}_p6ext.json") if (run_dir / n).exists()), None)
    split = ({"fold_assignments_file": folds.name, "sha256": _sha_file(folds), "protocol_id": man.get("protocol_id"), "protocol_sha256": man.get("protocol_sha256"),
              "seeds": man.get("seeds")} if folds.exists() else None)
    perf = ({"status": "EVALUATED_BY_REGISTERED_RUN", "results_file": res.name, "results_sha256": _sha_file(res), "experiment_manifest_sha256": _sha_file(mp),
             "evidence_status": exp.get("evidence_status"), "uncertainty": "cluster bootstrap intervals are inside the results file"} if res else None)
    unrec = {}
    if split is None:
        unrec["split_identifiers"] = "fold assignment file missing from the validation run"
    if perf is None:
        unrec["performance"] = "results file missing from the validation run"
    return split, perf, unrec


def build_training_provenance(*, kind: str, stem: str, model_version: str, features: list[str], dataset_info: dict, manifest_status: dict,
                              protocol_version: str, seed: int, training_date: str, X, validation_pointer: str,
                              threshold: float, threshold_rationale: str, artifact_sha256: str, code_revision: str | None,
                              validation_run_dir: Path | None = None) -> dict:
    tgt, hor = _task_text(kind)
    split, perf, unrec = validation_link(validation_run_dir, kind)
    return {
        "contract_version": CONTRACT_VERSION, "model_id": stem, "model_version": model_version,
        "dataset_manifest": {"dataset_sha256": dataset_info.get("sha256"), "manifest_status": manifest_status,
                             "rows": dataset_info.get("n_rows"), "subjects": dataset_info.get("n_subjects")},
        "protocol_version": protocol_version, "training_code_revision": code_revision, "runtime_versions": runtime_versions(),
        "feature_schema": {"order": list(features), "count": len(features)}, "feature_units": feature_units(list(features)),
        "preprocessing": {"basis": "training-matrix moments computed directly from X (ddof=0); the pipeline's own scaler, if any, is fitted on the same matrix",
                          **training_matrix_moments(X)},
        "target_definition": tgt, "prediction_horizon": hor, "training_date": training_date, "random_seeds": {"training_seed": seed},
        "validation_pointer": validation_pointer,
        "calibration": {"calibrated": False, "parameters": None,
                        "note": "scores of class-weighted / tree models are not probabilities; no recalibration has been fitted"},
        "decision_threshold": {"value": threshold, "rationale": threshold_rationale},
        "known_limitations": [K.DEMO_ONLY_LIMITATION, "No external validation. Small number of independent subjects.",
                              "Scores are uncalibrated."] + ([K.TEST_TIME_MISMATCH_NOTE] if kind == "risk" else []),
        "artifact_checksum": {"file": f"{stem}.joblib", "sha256": artifact_sha256},
        "intended_use": intended_use(), "split_identifiers": split, "performance": perf, "unrecoverable": unrec,
    }


def legacy_provenance(kind: str, models_dir: Path = MODELS) -> dict:
    """Provenance for the artifacts shipped before the contract. Every value is read from the artifact, its metadata, MANIFEST.json or
    the (superseded) training script; what none of them contain is declared unrecoverable."""
    stem = F.STEMS[kind]
    model, meta = F.load_artifact(kind, models_dir)               # refuses unless SHA-256 matches MANIFEST.json
    man = json.loads((Path(models_dir) / "MANIFEST.json").read_text())
    sc = F.find_scaler(model)
    feats = list(meta["feature_columns_in_order"])
    tgt, hor = _task_text(kind)
    script = "ml/training/train_parkinsons_classifier.py" if kind == "disease" else "ml/training/train_risk_classifier.py"
    added = _git("log", "--diff-filter=A", "--format=%H", "--", f":(top)backend/ml/models/{stem}.joblib")
    seed_note = ("estimator random_state=42 appears in the superseded training script; the cross-validation splitters (LeaveOneGroupOut / "
                 "GroupKFold) are deterministic. Read from source code, not recorded in the artifact.")
    return {
        "contract_version": CONTRACT_VERSION, "model_id": stem, "model_version": "legacy-unversioned",
        "dataset_manifest": {"dataset_sha256": None, "manifest_status": "UNPINNED_NO_ACCESS (ml/data/DATASET_MANIFEST.json)",
                             "rows": int(sc.n_samples_seen_), "subjects": meta.get("n_subjects")},
        "protocol_version": None, "training_code_revision": None,
        "runtime_versions": {"trained_with_scikit-learn": man.get("trained_with_sklearn"),
                             "trained_with_scikit-learn_basis": "inferred from the InconsistentVersionWarning raised on unpickling (MANIFEST.json)"},
        "feature_schema": {"order": feats, "count": len(feats)}, "feature_units": feature_units(feats),
        "preprocessing": {"scaler": "StandardScaler", "n_samples_seen": int(sc.n_samples_seen_),
                          "mean": [float(v) for v in sc.mean_], "scale": [float(v) for v in sc.scale_]},
        "target_definition": tgt, "prediction_horizon": hor, "training_date": None,
        "random_seeds": {"estimator_random_state": 42, "basis": seed_note},
        "validation_pointer": {"metadata_key": "loso_metrics" if kind == "disease" else "pooled_metrics",
                               "caveat": "historical, model selection on the scoring folds, no confidence interval, not reproduced from data",
                               "registered_protocol_results": "NOT_RUN (data inaccessible): ml/validation/results/phase4a/*_NOT_RUN.json"},
        "calibration": {"calibrated": False, "parameters": None, "note": "class-weighted / tree scores; no recalibration fitted"},
        "decision_threshold": {"value": 0.5, "rationale": None,
                               "note": "the estimator's own predict() rule (about 0.5 on predict_proba); no rationale was recorded"},
        "known_limitations": [K.DEMO_ONLY_LIMITATION, "No external validation.", "Scores are uncalibrated.",
                              "Model selection used the same folds that produced the reported score (optimistic)."]
                             + ([K.TEST_TIME_MISMATCH_NOTE] if kind == "risk" else []),
        "artifact_checksum": {"file": f"{stem}.joblib", "sha256": man["sha256"][f"{stem}.joblib"], "verified_against_manifest": True},
        "intended_use": intended_use(),
        "split_identifiers": None,
        "performance": {"status": "HISTORICAL_UNVERIFIED_NOT_REPRODUCED", "uncertainty": None,
                        "source": "metadata key " + ("loso_metrics" if kind == "disease" else "pooled_metrics") + " (see validation_pointer)"},
        "training_script_superseded": script, "artifact_first_committed_in": added.splitlines()[-1] if added else None,
        "unrecoverable": {
            "dataset_manifest": "the dataset file was never pinned; no SHA-256 of the training data is recorded anywhere",
            "protocol_version": "the artifact predates the registered protocols",
            "training_code_revision": "no git revision was recorded at training time (artifact_first_committed_in is the commit that added the file, not necessarily the training revision)",
            "training_date": "not recorded in the artifact or its metadata",
            "decision_threshold.rationale": "not recorded",
            "split_identifiers": "no fold assignment was saved; the splitters (LeaveOneGroupOut / GroupKFold) are deterministic but the file they split is unpinned",
            "performance": "historical figures only, with no interval and selected on the scoring folds; the registered re-evaluation is NOT_RUN",
        },
    }


def validate_provenance(prov: dict) -> dict:
    unrec = dict(prov.get("unrecoverable") or {})
    covered = {k.split(".")[0] for k in unrec}
    missing, empty = [], []
    for f in REQUIRED_FIELDS:
        if f not in prov:
            missing.append(f)
            continue
        v = prov[f]
        has_value = v not in (None, "", [], {}) and not (isinstance(v, dict) and v and all(x is None for x in v.values()))
        if not has_value and f not in covered:
            empty.append(f)
    # a declared-unrecoverable field must really be empty/absent-valued somewhere, otherwise the declaration is misleading
    if missing or empty:
        status = INVALID
    elif unrec:
        status = INCOMPLETE_DECLARED
    else:
        status = COMPLETE
    sha = (prov.get("artifact_checksum") or {}).get("sha256")
    if status != INVALID and not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        status = INVALID
        missing.append("artifact_checksum.sha256")
    return {"status": status, "missing": missing, "empty_without_declaration": empty, "unrecoverable": unrec,
            "contract_version": prov.get("contract_version")}


def sidecar_path(kind: str, models_dir: Path = MODELS) -> Path:
    return Path(models_dir) / f"{F.STEMS[kind]}_provenance.json"


def write_legacy_sidecars(models_dir: Path = MODELS) -> list[Path]:
    out = []
    for kind in ("disease", "risk"):
        p = sidecar_path(kind, models_dir)
        p.write_text(json.dumps(legacy_provenance(kind, models_dir), indent=2) + "\n")
        out.append(p)
    return out


def load_sidecar_report(kind: str, models_dir: Path = MODELS) -> dict:
    """Used by the API info endpoints. Never raises: a missing/corrupt sidecar is reported as INVALID."""
    p = sidecar_path(kind, models_dir)
    try:
        return validate_provenance(json.loads(p.read_text()))
    except Exception as e:  # noqa: BLE001
        return {"status": INVALID, "missing": list(REQUIRED_FIELDS), "reason": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    for p in write_legacy_sidecars():
        print(p, validate_provenance(json.loads(p.read_text()))["status"])
