"""
Reproducible final-model training (Phase 2, B4). NOT RUN in the audit sandbox (the UCI files are unreachable).

    cd backend
    python -m ml.validation.validate_models fetch
    python -m ml.validation.validate_models disease      # first: honest nested estimate of the PROCEDURE
    python -m ml.training.train_final disease            # then: fit the final model with the same procedure
    python -m ml.training.train_final risk

Procedure: the candidate is selected by group K-fold on ALL training patients (the same inner procedure whose
performance the nested run estimates), refit on all patients, and saved together with
  * <stem>.joblib                 pickle  (loading executes code -> SHA-256 pinned in MANIFEST.json)
  * <stem>_metadata.json          provenance: dataset sha256, sklearn/numpy/python versions, git commit, seed,
                                  feature order, candidate table, pointer to the nested-validation results
  * <stem>_domain.json            fitted input domain (ranges + shrinkage-Mahalanobis, LOGO-calibrated), no pickle
Written to ml/models/v2/ by default so shipped artifacts are never overwritten silently.

The metadata deliberately contains NO performance numbers computed on the training data and NO claim of clinical
validity; it points to `ml/validation/results/*.json` for the (nested) estimate.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.ood import fit_domain
from ml.evalkit import inner_select
from ml.validation import candidates as C
from ml.validation import data as D

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def train(stem: str, kind: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series, cands: dict, *, dataset_info: dict,
          out_dir: Path, seed: int, inner_k: int = 5, results_pointer: str | None = None) -> dict:
    import sklearn
    X, y, g = X.reset_index(drop=True), y.reset_index(drop=True), pd.Series(np.asarray(groups)).reset_index(drop=True)
    name, scores, _, status = inner_select(cands, X, y, g, inner_k=inner_k, seed=seed)
    model = cands[name]()
    model.fit(X, y)
    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / f"{stem}.joblib"
    joblib.dump(model, jl)
    domain = fit_domain(kind, X, groups=g)
    (out_dir / f"{stem}_domain.json").write_text(domain.to_json() + "\n")
    meta = {
        "model": f"{name} (scikit-learn Pipeline)", "selection": {"procedure": f"StratifiedGroupKFold({inner_k}) over all patients",
                                                                    "candidate_scores_roc_auc_inner_oof": scores, "status": status},
        "feature_columns_in_order": list(X.columns), "n_rows": int(len(X)), "n_groups": int(g.nunique()),
        "class_balance_rows": {int(k): int(v) for k, v in y.value_counts().items()},
        "dataset": dataset_info, "seed": seed, "git_commit": _git_commit(), "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__, "python_version": platform.python_version(),
        "validation": {"estimate_of_procedure": results_pointer or "RUN `python -m ml.validation.validate_models` AND LINK ITS JSON HERE",
                       "performance_numbers_in_this_file": "none (by design)", "clinically_validated": False},
        "domain_file": f"{stem}_domain.json",
        "disclaimer": "Demonstration model on a small public research dataset. Clinically unvalidated; not for patient care.",
    }
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["disease", "risk"])
    ap.add_argument("--csv")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--out-dir", default=str(MODELS / "v2"))
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    if a.which == "disease":
        df, info = D.load_disease(Path(a.csv) if a.csv else D.fetch("parkinsons"))
        meta = train("parkinsons_demo_classifier", "voice", df[D.DISEASE_FEATURES], df["status"], df["subject"],
                     C.disease_candidates(a.fast), dataset_info=info, out_dir=Path(a.out_dir), seed=a.seed,
                     results_pointer="ml/validation/results/disease.json")
    else:
        df, dcols, info = D.load_risk(Path(a.csv) if a.csv else D.fetch("parkinsons_updrs"))
        cols = dcols + ["test_time", "baseline_total_UPDRS"]
        meta = train("parkinsons_risk_classifier", "risk", df[cols], df["worsening"], df["subject#"].astype(str),
                     C.risk_candidates(a.fast), dataset_info=info, out_dir=Path(a.out_dir), seed=a.seed,
                     results_pointer="ml/validation/results/risk.json")
    print(json.dumps({k: meta[k] for k in ("model", "selection", "sklearn_version")}, indent=2))
    print("Next: python -m ml.validation.validate_models manifest   (after copying artifacts into ml/models/ deliberately)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
