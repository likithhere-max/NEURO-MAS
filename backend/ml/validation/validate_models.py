"""
Independent, leakage-safe validation of the two NEURO-MAS demonstration ML models (Phase 2).

    cd backend
    python -m ml.validation.validate_models fetch                    # download + hash the 2 UCI files (needs internet)
    python -m ml.validation.validate_models disease [--csv FILE]     # nested LOSO, subject-level metrics
    python -m ml.validation.validate_models risk    [--csv FILE]     # nested GroupKFold + ablations
    python -m ml.validation.validate_models artifacts                # sha256/version/feature-count checks
    python -m ml.validation.validate_models synthetic-smoke          # code-path check on FAKE data

Every JSON written has: status (completed | failed | not_run), evidence_status, clinically_validated=false.
`synthetic-smoke` output carries evidence_status="SYNTHETIC_FIXTURE_NOT_EVIDENCE" and must never be quoted.
Procedure: see docs/phase2/ML_EXPERIMENT_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evalkit import (aggregate_by_group, auc_stat, cluster_bootstrap, evaluate_binary,
                        group_label_permutation_test, leave_one_group_out, naive_selection_oof, nested_group_cv,
                        paired_cluster_bootstrap_diff, ppv_npv_at_prevalence, reliability_table, threshold_metrics)
from ml.validation import candidates as C
from ml.validation import data as D
from ml.validation import fingerprint as F

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODELS = HERE.parent / "models"
PREVALENCES = (0.005, 0.01, 0.05, 0.2, 0.75)
warnings.filterwarnings("ignore")


def env_info() -> dict:
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "platform": platform.platform()}


def envelope(task: str, seed: int, evidence_status: str, **body) -> dict:
    return {"task": task, "status": "completed", "evidence_status": evidence_status, "clinically_validated": False,
            "seed": seed, "environment": env_info(), **body}


def write(name: str, payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return p


# ----------------------------------------------------------------------------- disease
def run_disease_df(df: pd.DataFrame, info: dict, *, seed: int, n_boot: int, n_perm: int, fast: bool, evidence: str) -> dict:
    t0 = time.time()
    X = df[D.DISEASE_FEATURES].reset_index(drop=True)
    y, g = df["status"].reset_index(drop=True), df["subject"].reset_index(drop=True)
    cands, bases = C.disease_candidates(fast), C.disease_baselines(fast)

    # 1. NESTED LOSO: outer = each subject held out once; inner group-5-fold selects among candidates on train only
    nested = nested_group_cv(X, y, g, cands, outer="logo", inner_k=5, seed=seed)
    res = pd.DataFrame({"subject": g, "y": y, "p": nested.oof})
    subj = aggregate_by_group(g, y, nested.oof, "mean").rename(columns={"group": "subject"})

    def eval_level(frame, label):
        m = evaluate_binary(frame["y"], frame["p"], 0.5)
        ci = cluster_bootstrap(frame.assign(_c=frame["subject"] if "subject" in frame else frame.index), "_c",
                               lambda d: auc_stat(d), n_boot=n_boot, seed=seed)
        return {"level": label, **m, "roc_auc_cluster_bootstrap_ci95": ci}

    rec_level, subj_level = eval_level(res, "recording"), eval_level(subj, "subject")
    sens, spec = subj_level["at_threshold"]["sensitivity"], subj_level["at_threshold"]["specificity"]
    ppv = ({str(p): ppv_npv_at_prevalence(sens, spec, p) for p in PREVALENCES}
           if sens is not None and spec is not None else None)

    # 2. the SAME nested protocol applied to the baselines (no-skill and a 3-feature classical model)
    baseline_out = {}
    for name, fac in bases.items():
        r = nested_group_cv(X, y, g, {name: fac}, outer="logo", inner_k=5, seed=seed)
        s = aggregate_by_group(g, y, r.oof, "mean").rename(columns={"group": "subject"})
        baseline_out[name] = {"subject_level": evaluate_binary(s["y"], s["p"], 0.5)}
    paired = None
    if "classic3_LR" in bases:
        r = nested_group_cv(X, y, g, {"classic3_LR": bases["classic3_LR"]}, outer="logo", inner_k=5, seed=seed)
        s2 = aggregate_by_group(g, y, r.oof, "mean").rename(columns={"group": "subject"})
        pair = subj.assign(p_base=s2["p"].to_numpy())
        paired = paired_cluster_bootstrap_diff(pair, "subject", "p", "p_base", n_boot=n_boot, seed=seed)

    # 3. the BIASED protocol, reported only to quantify the optimism of the old procedure
    naive = naive_selection_oof(cands, X, y, g)
    naive_auc = {k: auc_stat(pd.DataFrame({"y": y, "p": v})) for k, v in naive.items()}
    best = max(naive_auc, key=lambda k: naive_auc[k])

    # 4. permutation test of a FIXED pipeline (modal nested choice): tests association, not the selection procedure
    modal = max(nested.chosen_counts(), key=nested.chosen_counts().get)

    def p_fn(yp):
        oof = np.zeros(len(yp))
        for tr, te in leave_one_group_out(g, X=X):
            m = cands[modal]()
            m.fit(X.iloc[tr], yp[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        return oof

    perm = group_label_permutation_test(g, y.to_numpy(), p_fn, naive_auc[modal], n_perm=n_perm, seed=seed)

    return envelope(
        "disease_voice_classifier", seed, evidence,
        dataset=info, protocol="nested LOSO (outer by subject; inner StratifiedGroupKFold(5)); preprocessing and feature "
        "selection inside pipelines; threshold fixed at 0.5 a priori",
        selection={"candidates": list(cands), "chosen_counts": nested.chosen_counts(),
                   "fallbacks": sum(f.selection_status != "selected" for f in nested.folds)},
        nested_performance={"recording_level": rec_level, "subject_level": subj_level,
                            "reliability_subject_level": reliability_table(subj["y"], subj["p"], 5),
                            "ppv_npv_at_prevalence": ppv},
        baselines=baseline_out, paired_auc_difference_vs_classic3={"definition": "AUC(nested pipeline) - AUC(classic3_LR), subject bootstrap", **(paired or {})},
        biased_protocol_for_comparison={"per_candidate_loso_auc_recording_level": naive_auc, "winner": best,
                                        "winner_auc": naive_auc[best],
                                        "note": "winner chosen on the same folds it is scored on -> optimistic"},
        permutation_test={"fixed_pipeline": modal, **perm},
        limitations=["~31-32 subjects; 8 healthy: wide intervals, sensitivity/specificity dominated by few people",
                     "single recording protocol, one lab; no external cohort",
                     "study prevalence ~75% PD is not a screening population (see PPV/NPV table)",
                     "clinically_validated is false regardless of the numbers"],
        runtime_seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- risk
def run_risk_df(df: pd.DataFrame, delta_cols: list[str], info: dict, *, seed: int, n_boot: int, fast: bool, evidence: str) -> dict:
    t0 = time.time()
    y, g = df["worsening"].reset_index(drop=True), df["subject#"].astype(str).reset_index(drop=True)
    sets = {
        "FULL (voice deltas + test_time + baseline_UPDRS)": delta_cols + ["test_time", "baseline_total_UPDRS"],
        "voice deltas ONLY": delta_cols,
        "test_time + baseline_UPDRS (NO voice)": ["test_time", "baseline_total_UPDRS"],
        "test_time ONLY": ["test_time"],
        "baseline_UPDRS ONLY": ["baseline_total_UPDRS"],
    }
    cands = C.risk_candidates(fast)
    preds, report = {}, {}
    for name, cols in sets.items():
        X = df[cols].reset_index(drop=True)
        r = nested_group_cv(X, y, g, cands, outer=10, inner_k=5, seed=seed, group_level_target=False)
        preds[name] = r.oof
        frame = pd.DataFrame({"subject": g, "y": y, "p": r.oof})
        report[name] = {**evaluate_binary(y, r.oof, 0.5), "chosen_counts": r.chosen_counts(),
                        "roc_auc_cluster_bootstrap_ci95": cluster_bootstrap(frame, "subject", auc_stat, n_boot=n_boot, seed=seed),
                        "reliability": reliability_table(y, r.oof, 5)}
    dummy = nested_group_cv(df[["test_time"]].reset_index(drop=True), y, g, {"dummy": lambda: __import__("sklearn.dummy").dummy.DummyClassifier(strategy="prior")},
                            outer=10, inner_k=3, seed=seed, group_level_target=False)
    full, novoice = list(sets)[0], "test_time + baseline_UPDRS (NO voice)"
    frame = pd.DataFrame({"subject": g, "y": y, "p_full": preds[full], "p_novoice": preds[novoice]})
    paired = paired_cluster_bootstrap_diff(frame, "subject", "p_full", "p_novoice", n_boot=n_boot, seed=seed)
    return envelope(
        "risk_concurrent_worsening_classifier", seed, evidence, dataset=info,
        protocol="nested GroupKFold(10) outer by patient; inner StratifiedGroupKFold(5); concurrent (same-visit) target; "
        "fixed threshold 0.5",
        ablation=report, dummy_prior=evaluate_binary(y, dummy.oof, 0.5),
        voice_incremental_value={"definition": "AUC(FULL) - AUC(test_time+baseline_UPDRS), patient bootstrap", **paired,
                                 "interpretation": "if the interval contains 0 the data cannot show that voice features add information"},
        label_note="worsening = total_UPDRS(t) - total_UPDRS(first visit) >= %.1f points: PROVISIONAL engineering threshold, "
                   "not a validated MCID" % info.get("worsening_threshold_points", 3.0),
        limitations=["target is concurrent, not a forecast of future worsening",
                     "labels vary within patient, so subject-level aggregation of the label is not defined; CIs resample patients",
                     "42 patients; one telemonitoring device; patients' repeated visits are autocorrelated",
                     "clinically_validated is false regardless of the numbers"],
        runtime_seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- synthetic fixtures (code-path checks ONLY)
def synthetic_disease(seed: int, n_subjects: int = 24, rec: int = 6) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subjects + 1):
        label = int(s <= round(n_subjects * 0.75))
        for r in range(1, rec + 1):
            row = {f: rng.normal(0, 1) + 0.7 * label * (i < 3) for i, f in enumerate(D.DISEASE_FEATURES)}
            rows.append({"name": f"phon_R01_S{s:02d}_{r}", "status": label, **row})
    df = pd.DataFrame(rows)
    info = D.verify_disease_frame(df, strict_counts=False)
    df["subject"] = df["name"].str.extract(r"(S\d+)_\d+$")[0]
    return df, {**info, "SYNTHETIC": True}


def synthetic_risk(seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, 16):
        base = rng.uniform(15, 40)
        slope = rng.normal(0.02, 0.02)
        for t in np.sort(rng.uniform(0, 180, 30)):
            rows.append({"subject#": s, "age": 60, "sex": 0, "test_time": t, "motor_UPDRS": 0, "total_UPDRS": base + slope * t + rng.normal(0, 1.5),
                         "Jitter(%)": rng.normal(0.006, 0.001), "HNR": rng.normal(22, 2)})
    raw = pd.DataFrame(rows)
    info = D.verify_risk_frame(raw, strict_counts=False)
    df, delta_cols = D.build_risk_frame(raw)
    info.update({"n_rows": len(df), "positive_rate": float(df["worsening"].mean()), "SYNTHETIC": True,
                 "worsening_threshold_points": D.WORSENING_THRESHOLD})
    return df, delta_cols, info


# ----------------------------------------------------------------------------- artifacts
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_artifacts() -> dict:
    import joblib
    import sklearn
    manifest = json.loads((MODELS / "MANIFEST.json").read_text())
    out = {"sklearn_runtime": sklearn.__version__, "artifacts": {}}
    for stem in ("parkinsons_demo_classifier", "parkinsons_risk_classifier"):
        jl, meta = MODELS / f"{stem}.joblib", json.loads((MODELS / f"{stem}_metadata.json").read_text())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = joblib.load(jl)
        n_in = getattr(model, "n_features_in_", None)
        out["artifacts"][stem] = {
            "sha256": sha256(jl), "manifest_match": manifest.get("sha256", {}).get(jl.name) == sha256(jl),
            "feature_count_matches_metadata": n_in == len(meta["feature_columns_in_order"]),
            "load_warnings": sorted({str(x.message)[:120] for x in w}),
            "metadata_records_sklearn_version": "sklearn_version" in meta, "metadata_records_dataset_hash": "dataset_sha256" in meta}
    return envelope("artifacts", 0, "REAL_RUN_ON_REPO_FILES", **out)


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "disease", "risk", "artifacts", "manifest", "synthetic-smoke"])
    ap.add_argument("--csv")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    try:
        if a.cmd == "fetch":
            for n in D.SOURCES:
                print(n, "->", D.fetch(n))
            return 0
        if a.cmd == "manifest":
            old = json.loads((MODELS / "MANIFEST.json").read_text()) if (MODELS / "MANIFEST.json").exists() else {}
            old["sha256"] = {p.name: sha256(p) for p in sorted(MODELS.glob("*.joblib"))}
            (MODELS / "MANIFEST.json").write_text(json.dumps(old, indent=2) + "\n")
            print(json.dumps(old["sha256"], indent=2))
            return 0
        if a.cmd == "artifacts":
            print(write("artifacts", run_artifacts(), out_dir))
            return 0
        if a.cmd == "synthetic-smoke":
            df, info = synthetic_disease(a.seed)
            r = run_disease_df(df, info, seed=a.seed, n_boot=30, n_perm=5, fast=True, evidence="SYNTHETIC_FIXTURE_NOT_EVIDENCE")
            rdf, dcols, rinfo = synthetic_risk(a.seed)
            r2 = run_risk_df(rdf, dcols, rinfo, seed=a.seed, n_boot=30, fast=True, evidence="SYNTHETIC_FIXTURE_NOT_EVIDENCE")
            print(write("synthetic_disease_NOT_EVIDENCE", r, out_dir), write("synthetic_risk_NOT_EVIDENCE", r2, out_dir))
            return 0
        if a.cmd == "disease":
            path = Path(a.csv) if a.csv else D.fetch("parkinsons")
            df, info = D.load_disease(path)
            # Phase 4A (E03): a file of the right SHAPE is not necessarily the real data. Only a file that reproduces the shipped
            # artifact's scaler fingerprint is labelled as authenticated; anything else is explicitly "unauthenticated".
            auth = F.authenticate("disease", path)
            res = run_disease_df(df, info, seed=a.seed, n_boot=a.n_boot, n_perm=a.n_perm, fast=a.fast, evidence=F.evidence_status(auth))
            print(write("disease", {**res, "authentication": auth}, out_dir))
        else:
            path = Path(a.csv) if a.csv else D.fetch("parkinsons_updrs")
            df, dcols, info = D.load_risk(path)
            auth = F.authenticate("risk", path)
            res = run_risk_df(df, dcols, info, seed=a.seed, n_boot=a.n_boot, fast=a.fast, evidence=F.evidence_status(auth))
            print(write("risk", {**res, "authentication": auth}, out_dir))
        return 0
    except Exception as e:  # a failed run is REPORTED as failed, never as a partial success
        payload = {"task": a.cmd, "status": "failed", "evidence_status": "NONE", "clinically_validated": False,
                   "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:], "environment": env_info()}
        print(write(f"{a.cmd}_FAILED", payload, out_dir), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
