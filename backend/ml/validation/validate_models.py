"""
Independent validation of the two NEURO-MAS demo ML models (audit deliverable: ML Validation Report).

WHAT THIS ADDS over ml/training/*.py (see docs/audit/ML_VALIDATION_REPORT.md for why each matters):
  disease : (1) subject-ID sanity checks, (2) NESTED subject-level model selection (the training script picks
            the winner on the same LOSO folds it then reports -> optimistic), (3) subject-level aggregation,
            (4) subject-level bootstrap 95% CI, (5) label-permutation test, (6) dummy baseline,
            (7) Brier score + ECE calibration, (8) PPV/NPV at realistic prevalences (the training set is ~75% PD).
  risk    : (1) ablation: time-only and baseline-UPDRS-only baselines vs the full model (the full model includes
            `test_time` and `baseline_total_UPDRS`, which are not voice features), (2) majority-class baseline,
            (3) subject-level bootstrap CI, (4) permutation test, (5) calibration.
  both    : artifact integrity (sha256, sklearn version, feature order) for the committed .joblib files.

USAGE (needs internet for the default download, or pass --csv):
  cd backend
  python ml/validation/validate_models.py disease [--csv parkinsons.data] [--n-boot 1000] [--n-perm 100] [--fast]
  python ml/validation/validate_models.py risk    [--csv parkinsons_updrs.data] [--n-boot 1000] [--n-perm 100]
  python ml/validation/validate_models.py artifacts
Results are written to ml/validation/results_<cmd>.json and printed. Everything is seeded (--seed, default 20261023).

HONESTY NOTE: nothing here makes the models clinically valid. The datasets are tiny research corpora
(31-32 voice-study subjects; 42 telemonitoring subjects) and these models are demonstration artifacts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
DISEASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
RISK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- statistics helpers
def auc_safe(y, p):
    y = np.asarray(y)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def ece(y, p, bins=10):
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def subject_bootstrap_ci(df, n_boot, rng, stat=auc_safe):
    """Resample SUBJECTS (not rows) with replacement -> 95% percentile CI. df has columns subject, y, p."""
    subs = df["subject"].unique()
    groups = {s: g for s, g in df.groupby("subject")}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        b = pd.concat([groups[s] for s in pick])
        v = stat(b["y"], b["p"])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return {"lo": None, "hi": None, "n_valid": 0}
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "n_valid": len(vals)}


def ppv_npv(sens, spec, prevalence):
    ppv = sens * prevalence / (sens * prevalence + (1 - spec) * (1 - prevalence))
    npv = spec * (1 - prevalence) / (spec * (1 - prevalence) + (1 - sens) * prevalence)
    return float(ppv), float(npv)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(csv: str | None, url: str, **kw) -> tuple[pd.DataFrame, str]:
    src = csv or url
    df = pd.read_csv(src, **kw)
    h = sha256(Path(csv)) if csv else hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    return df, h


# ----------------------------------------------------------------------------- candidates
def candidates(fast: bool):
    n = 60 if fast else 200
    return {
        "RandomForest": lambda: Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=42, n_jobs=-1))]),
        "LogisticRegression": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5, random_state=42))]),
        "SVM_RBF": lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))]),
        "RandomForest_Top10": lambda: Pipeline([("s", StandardScaler()), ("k", SelectKBest(f_classif, k=10)), ("c", RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=42, n_jobs=-1))]),
    }


def proba1(model, X):
    return model.predict_proba(X)[:, 1]


def loso_oof(factory, X, y, groups):
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = factory(); m.fit(X.iloc[tr], y.iloc[tr]); oof[te] = proba1(m, X.iloc[te])
    return oof


def inner_select(cands, X, y, groups, k=5, seed=0):
    """Pick the candidate with best inner GroupKFold ROC-AUC using ONLY the given (outer-train) data."""
    n_splits = min(k, groups.nunique())
    best, best_auc = None, -1
    for name, factory in cands.items():
        oof = np.zeros(len(y)); ok = True
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            if y.iloc[tr].nunique() < 2:
                ok = False; break
            m = factory(); m.fit(X.iloc[tr], y.iloc[tr]); oof[te] = proba1(m, X.iloc[te])
        a = auc_safe(y, oof) if ok else -1
        if a > best_auc:
            best, best_auc = name, a
    return best


# ----------------------------------------------------------------------------- disease model
def run_disease(args):
    rng = np.random.default_rng(args.seed)
    df, data_hash = load_csv(args.csv, DISEASE_URL)
    df["subject"] = df["name"].str.split("_").str[2]
    feats = [c for c in df.columns if c not in ("name", "status", "subject")]
    X, y, g = df[feats].reset_index(drop=True), df["status"].reset_index(drop=True), df["subject"].reset_index(drop=True)

    sub_status = df.groupby("subject")["status"].agg(["nunique", "first", "size"])
    report = {
        "dataset_sha256": data_hash, "n_rows": int(len(df)), "n_features": len(feats),
        "n_unique_subject_ids": int(g.nunique()),
        "subjects_with_mixed_labels": sub_status.index[sub_status["nunique"] > 1].tolist(),
        "recordings_per_subject": {"min": int(sub_status["size"].min()), "max": int(sub_status["size"].max())},
        "subjects_pd": int((sub_status["first"] == 1).sum()), "subjects_healthy": int((sub_status["first"] == 0).sum()),
        "row_prevalence_pd": float(y.mean()),
        "note_subject_count": "UCI documents 31 people; this file's ID parse yields the count above. A mismatch means "
                               "either an ID-parsing artefact (same person under two IDs -> LOSO leakage) or a documentation discrepancy.",
        "seed": args.seed,
    }
    print(json.dumps({k: report[k] for k in ("n_rows", "n_unique_subject_ids", "subjects_pd", "subjects_healthy", "subjects_with_mixed_labels")}, indent=1))

    cands = candidates(args.fast)

    # (a) the training script's protocol: pick winner on the SAME LOSO folds it reports (optimistic)
    print("naive protocol (select on reported folds) ...")
    naive = {n: loso_oof(f, X, y, g) for n, f in cands.items()}
    naive_auc = {n: auc_safe(y, p) for n, p in naive.items()}
    naive_best = max(naive_auc, key=naive_auc.get)
    report["naive_selection"] = {"per_model_roc_auc_recording_level": naive_auc, "winner": naive_best, "winner_auc": naive_auc[naive_best]}

    # (b) NESTED: outer LOSO; inner GroupKFold selects the model using only that outer-train split
    print("nested protocol (inner selection per outer fold) ...")
    nested = np.zeros(len(y)); chosen = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        name = inner_select(cands, X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True), g.iloc[tr].reset_index(drop=True))
        chosen.append(name)
        m = cands[name](); m.fit(X.iloc[tr], y.iloc[tr]); nested[te] = proba1(m, X.iloc[te])
    res = pd.DataFrame({"subject": g, "y": y, "p": nested})
    subj = res.groupby("subject").agg(y=("y", "first"), p=("p", "mean")).reset_index()
    report["nested_loso"] = {
        "model_choice_counts": pd.Series(chosen).value_counts().to_dict(),
        "recording_level_roc_auc": auc_safe(res["y"], res["p"]),
        "subject_level_roc_auc": auc_safe(subj["y"], subj["p"]),
        "subject_level_ci95": subject_bootstrap_ci(subj, args.n_boot, rng),
        "recording_level_ci95_subject_bootstrap": subject_bootstrap_ci(res, args.n_boot, rng),
        "brier": float(brier_score_loss(res["y"], res["p"])), "ece": ece(res["y"], res["p"]),
        "brier_of_constant_prevalence_prediction": float(brier_score_loss(res["y"], np.full(len(res), res["y"].mean()))),
    }
    pred = (res["p"] >= 0.5).astype(int)
    sens = float(((pred == 1) & (res["y"] == 1)).sum() / max((res["y"] == 1).sum(), 1))
    spec = float(((pred == 0) & (res["y"] == 0)).sum() / max((res["y"] == 0).sum(), 1))
    report["nested_loso"].update({"sensitivity_at_0.5": sens, "specificity_at_0.5": spec,
        "ppv_npv_by_prevalence": {str(pr): dict(zip(("ppv", "npv"), ppv_npv(sens, spec, pr))) for pr in (0.01, 0.05, 0.2, 0.75)}})

    # (c) baselines
    dummy = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        d = DummyClassifier(strategy="prior").fit(X.iloc[tr], y.iloc[tr]); dummy[te] = d.predict_proba(X.iloc[te])[:, 1]
    report["baselines"] = {"dummy_prior_roc_auc": auc_safe(y, dummy), "majority_class_accuracy": float(max(y.mean(), 1 - y.mean()))}

    # (d) permutation test on the fixed naive-winner pipeline (labels shuffled at SUBJECT level)
    print(f"permutation test ({args.n_perm} shuffles) ...")
    obs = naive_auc[naive_best]
    subj_lab = df.groupby("subject")["status"].first()
    perm_aucs = []
    for _ in range(args.n_perm):
        shuffled = pd.Series(rng.permutation(subj_lab.values), index=subj_lab.index)
        yp = g.map(shuffled).astype(int)
        perm_aucs.append(auc_safe(yp, loso_oof(cands[naive_best], X, yp, g)))
    report["permutation_test"] = {"model": naive_best, "observed_auc": obs, "n_perm": args.n_perm,
        "perm_auc_mean": float(np.nanmean(perm_aucs)), "p_value": float((1 + np.sum(np.array(perm_aucs) >= obs)) / (1 + args.n_perm))}
    out = HERE / "results_disease.json"; out.write_text(json.dumps(report, indent=2, default=str)); print(json.dumps(report["nested_loso"], indent=1, default=str)); print("->", out)


# ----------------------------------------------------------------------------- risk model
def run_risk(args):
    rng = np.random.default_rng(args.seed)
    df, data_hash = load_csv(args.csv, RISK_URL)
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values(["subject#", "test_time"]).reset_index(drop=True)
    voice = [c for c in df.columns if c not in ("subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS")]
    base = df.groupby("subject#", as_index=False).first().set_index("subject#")
    df["baseline_total_UPDRS"] = df["subject#"].map(base["total_UPDRS"])
    for c in voice:
        df[f"baseline_{c}"] = df["subject#"].map(base[c])
    df = df[df.groupby("subject#")["test_time"].transform("min") != df["test_time"]].reset_index(drop=True)
    df["worsening"] = ((df["total_UPDRS"] - df["baseline_total_UPDRS"]) >= 3.0).astype(int)
    for c in voice:
        df[f"delta_{c}"] = df[c] - df[f"baseline_{c}"]
    delta_cols = [f"delta_{c}" for c in voice]
    y, g = df["worsening"].reset_index(drop=True), df["subject#"].reset_index(drop=True)

    sets = {
        "FULL (voice deltas + test_time + baseline_UPDRS)  [= saved model]": delta_cols + ["test_time", "baseline_total_UPDRS"],
        "voice deltas ONLY": delta_cols,
        "test_time ONLY": ["test_time"],
        "baseline_total_UPDRS ONLY": ["baseline_total_UPDRS"],
        "test_time + baseline_UPDRS (NO voice)": ["test_time", "baseline_total_UPDRS"],
    }
    mk = lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5, random_state=42))])
    report = {"dataset_sha256": data_hash, "n_rows": int(len(df)), "n_subjects": int(g.nunique()), "positive_rate": float(y.mean()),
              "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())), "seed": args.seed, "ablation": {}}
    oof_full = None
    for name, cols in sets.items():
        X = df[cols].reset_index(drop=True); oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=10).split(X, y, g):
            m = mk(); m.fit(X.iloc[tr], y.iloc[tr]); oof[te] = proba1(m, X.iloc[te])
        res = pd.DataFrame({"subject": g, "y": y, "p": oof})
        report["ablation"][name] = {"roc_auc": auc_safe(y, oof), "ci95_subject_bootstrap": subject_bootstrap_ci(res, args.n_boot, rng),
                                    "brier": float(brier_score_loss(y, oof)), "ece": ece(y, oof),
                                    "accuracy_at_0.5": float(((oof >= .5).astype(int) == y).mean())}
        print(f"{name:70s} AUC={report['ablation'][name]['roc_auc']:.3f}  CI={report['ablation'][name]['ci95_subject_bootstrap']}")
        if name.startswith("FULL"):
            oof_full = oof
    # permutation test on the FULL model (shuffle worsening label across subjects' rows is invalid -> shuffle within-subject-block: permute subject-level label profiles)
    subs = g.unique(); prof = {s: y[g == s].values for s in subs}
    X = df[sets[next(iter(sets))]].reset_index(drop=True); obs = report["ablation"][next(iter(sets))]["roc_auc"]; perm = []
    for _ in range(args.n_perm):
        order = rng.permutation(subs); yp = y.copy()
        for s, t in zip(subs, order):
            v = prof[t]; idx = np.where(g == s)[0]
            yp.iloc[idx] = np.resize(v, len(idx))
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=10).split(X, yp, g):
            m = mk(); m.fit(X.iloc[tr], yp.iloc[tr]); oof[te] = proba1(m, X.iloc[te])
        perm.append(auc_safe(yp, oof))
    report["permutation_test_full_model"] = {"observed_auc": obs, "n_perm": args.n_perm, "perm_auc_mean": float(np.nanmean(perm)),
                                               "p_value": float((1 + np.sum(np.array(perm) >= obs)) / (1 + args.n_perm))}
    out = HERE / "results_risk.json"; out.write_text(json.dumps(report, indent=2, default=str)); print("->", out)


# ----------------------------------------------------------------------------- artifacts
def run_manifest(args):
    """Regenerate ml/models/MANIFEST.json after (re)training. Review the diff before committing."""
    man = {p.name: sha256(p) for p in sorted(MODELS.glob("*.joblib"))}
    (MODELS / "MANIFEST.json").write_text(json.dumps({"_comment": json.loads((MODELS / "MANIFEST.json").read_text()).get("_comment", "") if (MODELS / "MANIFEST.json").exists() else "", "sha256": man}, indent=2) + "\n")
    print(json.dumps(man, indent=2))


def run_artifacts(args):
    import joblib, sklearn
    out = {"sklearn_runtime": sklearn.__version__, "python": sys.version.split()[0], "artifacts": {}}
    for stem in ("parkinsons_demo_classifier", "parkinsons_risk_classifier"):
        jl, meta_p = MODELS / f"{stem}.joblib", MODELS / f"{stem}_metadata.json"
        meta = json.loads(meta_p.read_text())
        info = {"sha256": sha256(jl), "bytes": jl.stat().st_size, "metadata_has_sklearn_version": any("sklearn" in k for k in meta),
                "metadata_has_artifact_hash": any("sha" in k or "hash" in k for k in meta)}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always"); model = joblib.load(jl)
        info["load_warnings"] = sorted({str(x.message)[:120] for x in w})
        info["pipeline_steps"] = [s for s, _ in model.steps]
        n_in = getattr(model, "n_features_in_", None)
        info["n_features_in_model"] = n_in; info["n_features_in_metadata"] = len(meta["feature_columns_in_order"])
        info["feature_count_matches"] = n_in == len(meta["feature_columns_in_order"])
        out["artifacts"][stem] = info
    print(json.dumps(out, indent=2)); (HERE / "results_artifacts.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["disease", "risk", "artifacts", "manifest"])
    ap.add_argument("--csv"); ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--fast", action="store_true", help="smaller forests (for smoke tests)")
    a = ap.parse_args()
    {"disease": run_disease, "risk": run_risk, "artifacts": run_artifacts, "manifest": run_manifest}[a.cmd](a)
