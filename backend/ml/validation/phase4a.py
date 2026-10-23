"""
Phase 4A -- rigorous, pre-specified ML validation of the two Parkinson's demonstration models.

    cd backend
    python -m ml.validation.phase4a artifacts                                   # needs NO data: audits the shipped artifacts
    python -m ml.validation.phase4a authenticate --kind risk --csv FILE         # does FILE reproduce the training matrix?
    python -m ml.validation.phase4a run --kind disease --csv FILE [--seeds 5]   # the full protocol (needs the real file)
    python -m ml.validation.phase4a run --kind risk    --csv FILE [--seeds 5]
    python -m ml.validation.phase4a selftest                                    # harness integrity on SYNTHETIC data (not evidence)

Data location. `--csv FILE`, else the environment variable NEUROMAS_PARKINSONS_CSV (disease) / NEUROMAS_PARKINSONS_UPDRS_CSV
(risk), else ml/data/cache/<published file name>. If none of these exists the run is reported as NOT_RUN (exit code 2) and
the JSON says so. NOTHING is ever generated, imputed or substituted: a missing file never becomes a result.

What `run` evaluates. The PROCEDURE "select a model on training data only, fit it, predict unseen people" -- not the saved
artifact. (The saved artifact was fitted on every row, so scoring it on the same file would be a training-set score.)

The protocol below is FROZEN and hashed (PROTOCOL_ID / protocol_sha256 are written into every output). It was fixed before any
real-data result existed in this repository. Changing a number here changes the hash, so a changed protocol cannot masquerade as the
registered one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.evalkit import (aggregate_by_group, auc_stat, cluster_bootstrap, evaluate_binary, group_label_permutation_test,
                        leave_one_group_out, nested_group_cv, paired_cluster_bootstrap_diff, ppv_npv_at_prevalence,
                        reliability_table, wilson_interval)
from ml.validation import candidates as C
from ml.validation import data as D
from ml.validation import fingerprint as F

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase4a"
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------------------------ frozen protocol
PROTOCOL = {
    "id": "p4a-1",
    "threshold": 0.5,                      # fixed a priori; never tuned on any evaluation data
    "inner_k": 5,
    "selection_metric": "roc_auc",         # inner-loop selection uses the outer-TRAINING rows only
    "seed_base": 20261023,
    "default_seeds": 5,
    "n_boot": 2000,
    "n_perm": 200,
    "disease": {"outer": "logo", "unit": "subject (parsed from the recording name)", "target": "status (0 healthy / 1 PD), group-level",
                "baselines": ["dummy_prior", "classic3_LR"]},
    "risk": {"outer": 10, "unit": "patient (subject#)", "worsening_threshold_points": D.WORSENING_THRESHOLD,
             "target": "total_UPDRS(t) - total_UPDRS(first visit) >= threshold, concurrent, row-level",
             "feature_sets": {
                 "FULL": "voice deltas + test_time + baseline_total_UPDRS",
                 "VOICE_ONLY": "voice deltas",
                 "TIME_BASELINE": "test_time + baseline_total_UPDRS",
                 "TIME_ONLY": "test_time",
                 "BASELINE_ONLY": "baseline_total_UPDRS",
                 "DEMOGRAPHIC_ONLY": "age + sex",
                 "TIME_DEMOGRAPHIC": "test_time + age + sex",
                 "TIME_BASELINE_DEMOGRAPHIC": "test_time + baseline_total_UPDRS + age + sex"}},
    "reference_for_voice_value": ["TIME_BASELINE", "TIME_BASELINE_DEMOGRAPHIC"],
    "interpretation_rule": "voice adds measurable information ONLY IF the 95% cluster-bootstrap interval of AUC(FULL) - AUC(reference) "
                           "excludes 0 for every reference set AND the effect is reproduced for every seed; otherwise the data cannot "
                           "show that it does. Neither outcome is evidence of clinical usefulness.",
}
PROTOCOL_SHA256 = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()

ENV_VARS = {"disease": "NEUROMAS_PARKINSONS_CSV", "risk": "NEUROMAS_PARKINSONS_UPDRS_CSV"}
FILENAMES = {"disease": "parkinsons.data", "risk": "parkinsons_updrs.data"}


class DataUnavailable(RuntimeError):
    """The required data file is not present. Callers must report NOT_RUN; they must never substitute other data."""


def resolve_data_path(kind: str, csv: str | None = None) -> Path:
    tried = []
    for cand in (csv, os.environ.get(ENV_VARS[kind]), str(D.CACHE / FILENAMES[kind])):
        if not cand:
            continue
        tried.append(cand)
        if Path(cand).is_file():
            return Path(cand)
    raise DataUnavailable(
        f"No data file for the '{kind}' task. Looked at: {tried or ['(nothing was specified)']}. Download "
        f"{D.SOURCES['parkinsons' if kind == 'disease' else 'parkinsons_updrs']} on a machine with internet access and pass it with "
        f"--csv, or set {ENV_VARS[kind]}. No synthetic or substitute data is used.")


def env_info() -> dict:
    import joblib
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__, "platform": platform.platform()}


def envelope(task: str, status: str, evidence: str, **body) -> dict:
    return {"task": task, "status": status, "evidence_status": evidence, "clinically_validated": False,
            "protocol_id": PROTOCOL["id"], "protocol_sha256": PROTOCOL_SHA256, "protocol": PROTOCOL,
            "environment": env_info(), **body}


def write(name: str, payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return p


# ------------------------------------------------------------------------------------------------ small statistics
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float] | None:
    """Exact binomial interval. Valid for k successes among n INDEPENDENT units (here: whole subjects)."""
    if n <= 0:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def reconstruct_confusion(accuracy: float, precision: float, recall: float, n: int, n_pos: int) -> dict:
    """Recover the integer confusion matrix implied by pooled accuracy/precision/recall and the class counts.
    `consistent` is True only if the integers reproduce ALL THREE reported figures to 1e-9 -- a strong internal-consistency check
    of the reported numbers (it does not prove where the data came from)."""
    tp = round(recall * n_pos)
    fn = n_pos - tp
    fp = round(tp / precision) - tp if precision > 0 else None
    if fp is None:
        return {"consistent": False, "reason": "precision is 0"}
    tn = n - n_pos - fp
    ok = (tp >= 0 and fn >= 0 and fp >= 0 and tn >= 0 and abs((tp + tn) / n - accuracy) < 1e-9
          and abs(tp / (tp + fp) - precision) < 1e-9 and abs(tp / n_pos - recall) < 1e-9)
    out = {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "consistent": bool(ok)}
    if ok:
        out.update(sensitivity=tp / n_pos, specificity=tn / (n - n_pos), accuracy=(tp + tn) / n,
                   majority_class_accuracy=max(n_pos, n - n_pos) / n, balanced_accuracy=0.5 * (tp / n_pos + tn / (n - n_pos)))
    return out


# ------------------------------------------------------------------------------------------------ artifact audit (no data)
PUBLISHER_FACTS = {
    "source_note": "Read from the publisher's dataset pages on 2026-10-23 by fetching archive.ics.uci.edu (page text summarised by a "
                   "tool, NOT a hash-verified download of the data). Quoted figures should be re-checked against the pages.",
    "parkinsons_telemonitoring_id189": {"url": "https://archive.ics.uci.edu/dataset/189", "doi": "10.24432/C5ZS3N",
        "instances": 5875, "subjects": 42, "missing_values": False, "license": "CC BY 4.0",
        "total_UPDRS": "Clinician's total UPDRS score, linearly interpolated", "motor_UPDRS": "Clinician's motor UPDRS score, linearly interpolated",
        "test_time": "Time since recruitment into the trial; the integer part is the number of days since recruitment"},
    "parkinsons_id174": {"url": "https://archive.ics.uci.edu/dataset/174", "doi": "10.24432/C59C74",
        "subjects_per_publisher": 31, "subjects_with_PD_per_publisher": 23, "recordings_text": "195", "instances_metadata_field": 197,
        "recordings_per_subject": "around six", "missing_values": False, "license": "CC BY 4.0"},
}


def artifact_audit(models_dir: Path = F.MODELS) -> dict:
    out = {"models": {}, "publisher_facts": PUBLISHER_FACTS}
    for kind, stem in F.STEMS.items():
        meta = json.loads((Path(models_dir) / f"{stem}_metadata.json").read_text())
        model, _ = F.load_artifact(kind, models_dir)
        fp = F.artifact_fingerprint(kind, models_dir)
        steps = [s[0] for s in model.steps]
        entry = {"sha256_matches_manifest": True, "pipeline_steps": steps, "n_features": len(fp["features"]),
                 "training_rows_recorded_in_scaler": fp["n"], "metadata_keys": sorted(meta)}
        entry["records_sklearn_version"] = "sklearn_version" in meta
        entry["records_dataset_sha256"] = "dataset_sha256" in meta
        entry["records_seed"] = "seed" in meta or "random_state" in meta
        if kind == "disease":
            n, n_pos = int(meta["n_recordings"]), 147
            cands = meta["candidates_compared"]
            entry["metadata_n_subjects"] = meta["n_subjects"]
            entry["publisher_n_subjects"] = PUBLISHER_FACTS["parkinsons_id174"]["subjects_per_publisher"]
            entry["n_recordings_claimed"] = n
            entry["rows_match_scaler"] = (n == fp["n"])
            sel = model.named_steps["select"]
            idx = np.where(sel.get_support())[0]
            entry["selected_features_final_fit_all_rows"] = [fp["features"][i] for i in idx]
        else:
            n = int(meta["n_recordings_after_dropping_baseline_visits"])
            n_pos = round(meta["positive_class_rate"] * n)
            cands = meta["candidates_compared"]
            entry["n_recordings_claimed"] = n
            entry["rows_match_scaler"] = (n == fp["n"])
            entry["n_subjects_claimed"] = meta["n_subjects"]
            entry["expected_rows_if_only_first_visits_dropped"] = 5875 - meta["n_subjects"]
            entry["rows_dropped_beyond_first_visit_per_subject"] = 5875 - meta["n_subjects"] - n
            lr = model.named_steps["classifier"]
            coef = dict(zip(fp["features"], map(float, lr.coef_[0])))
            entry["logistic_coefficients_standardised"] = coef
            entry["abs_coef_test_time"] = abs(coef["test_time"])
            entry["abs_coef_baseline_total_UPDRS"] = abs(coef["baseline_total_UPDRS"])
            jit = {k: v for k, v in coef.items() if k.startswith("delta_Jitter")}
            entry["jitter_delta_coefficients"] = jit
            entry["jitter_coefficients_have_mixed_signs"] = (min(jit.values()) < 0 < max(jit.values()))
        entry["n_positive_rows_used"] = n_pos
        entry["reconstructed_confusion_matrices"] = {
            name: reconstruct_confusion(m["accuracy"], m["precision"], m["recall"], n, n_pos) for name, m in cands.items()}
        winner = meta["model"].split(" ")[0]
        wm = cands.get(winner) or meta.get("pooled_metrics") or meta.get("loso_metrics")
        entry["reported_winner"] = winner
        entry["reported_winner_metrics"] = {k: float(v) for k, v in wm.items()}
        rc = entry["reconstructed_confusion_matrices"].get(winner)
        if rc and rc.get("consistent"):
            tp, fp_, tn, fn = rc["TP"], rc["FP"], rc["TN"], rc["FN"]
            entry["winner_row_level_specificity_wilson95_NAIVE_LOWER_BOUND_ON_WIDTH"] = wilson_interval(tn, tn + fp_)
            entry["winner_row_level_sensitivity_wilson95_NAIVE_LOWER_BOUND_ON_WIDTH"] = wilson_interval(tp, tp + fn)
            entry["winner_accuracy_minus_majority_accuracy"] = rc["accuracy"] - rc["majority_class_accuracy"]
        out["models"][kind] = entry
    out["healthy_subject_specificity_resolution"] = {
        "note": "If the disease model's 8 healthy subjects each count as one independent unit, the exact 95% interval for the "
                "subject-level specificity is determined by k = number of healthy subjects classified correctly:",
        "clopper_pearson_n8": {str(k): clopper_pearson(k, 8) for k in range(9)}}
    return out


# ------------------------------------------------------------------------------------------------ metrics helpers
def _prauc(d: pd.DataFrame):
    return None if d["y"].nunique() < 2 else float(average_precision_score(d["y"], d["p"]))


def _sens(d: pd.DataFrame, thr: float):
    pos = d[d["y"] == 1]
    return None if len(pos) == 0 else float((pos["p"] >= thr).mean())


def _spec(d: pd.DataFrame, thr: float):
    neg = d[d["y"] == 0]
    return None if len(neg) == 0 else float((neg["p"] < thr).mean())


def _brier_skill(d: pd.DataFrame):
    p0 = d["y"].mean()
    ref = float(((p0 - d["y"]) ** 2).mean())
    return None if ref == 0 else float(1 - ((d["p"] - d["y"]) ** 2).mean() / ref)


def per_fold_auc(res, groups: pd.Series, y: pd.Series) -> list:
    out = []
    for f in res.folds:
        m = groups.isin(f.test_groups).to_numpy()
        out.append(None if y[m].nunique() < 2 else float(roc_auc_score(y[m], res.oof[m])))
    return out


def within_subject_auc(frame: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Mean over patients of the AUC computed inside each patient's own visits (does the score rank that patient's later/worse visits
    higher?). Patients with a single class are excluded and counted. Interval: bootstrap over patients."""
    aucs = []
    for _, d in frame.groupby("subject", sort=True):
        aucs.append(np.nan if d["y"].nunique() < 2 else float(roc_auc_score(d["y"], d["p"])))
    a = np.array(aucs, float)
    ok = a[~np.isnan(a)]
    out = {"n_patients": int(len(a)), "n_patients_with_both_classes": int(len(ok)), "n_excluded": int(len(a) - len(ok))}
    if len(ok) < 5:
        return {**out, "mean_auc": None, "note": "fewer than 5 patients with both classes: no interval"}
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(ok, size=len(ok), replace=True).mean()) for _ in range(n_boot)]
    return {**out, "mean_auc": float(ok.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "chance": 0.5}


def full_metrics(frame: pd.DataFrame, thr: float, n_boot: int, seed: int) -> dict:
    """frame columns: subject, y, p.  Row-level metrics with patient/subject-cluster bootstrap intervals."""
    ev = evaluate_binary(frame["y"], frame["p"], thr)
    ci = lambda stat: cluster_bootstrap(frame, "subject", stat, n_boot=n_boot, seed=seed)
    return {**ev,
            "roc_auc_cluster_bootstrap_ci95": ci(auc_stat),
            "pr_auc_cluster_bootstrap_ci95": ci(_prauc),
            "sensitivity_cluster_bootstrap_ci95": ci(lambda d: _sens(d, thr)),
            "specificity_cluster_bootstrap_ci95": ci(lambda d: _spec(d, thr)),
            "brier_skill_vs_constant_prevalence": _brier_skill(frame),
            "calibration_note": "scores come from class_weight='balanced' / unweighted-tree models and are NOT probabilities; the "
                                "reliability table is a diagnostic, not a calibration claim",
            "reliability_diagnostic": reliability_table(frame["y"], frame["p"], 5)}


def noskill_metrics(y, oof, thr: float) -> dict:
    """Metrics for the constant-prior (no-skill) baseline.

    Under leave-one-subject-out a constant-prior model scores each held-out subject with the prevalence of the OTHER subjects, which is
    lower for a held-out positive than for a held-out negative; pooled across folds that anti-correlates score with label and gives
    AUC ~ 0 for a model that has no information at all (the extreme case of the pooled-LOSO bias to below 0.5 documented in Phase 3, S4).
    The honest no-skill ranking metrics are those of a constant score (AUC 0.5, PR-AUC = prevalence); the raw LOSO value is kept, labelled."""
    raw = evaluate_binary(y, oof, thr)
    const = evaluate_binary(y, np.full(len(y), float(np.mean(y))), thr)
    out = dict(raw)
    for k in ("roc_auc", "pr_auc", "brier", "ece"):
        out[k] = const[k]
    out["loso_prior_artifact"] = {"roc_auc": raw["roc_auc"], "pr_auc": raw["pr_auc"],
                                  "note": "AUC of the out-of-fold constant-prior scores; an artifact of leaving a subject out, NOT a skill measure"}
    return out


# ------------------------------------------------------------------------------------------------ the two protocols
def run_risk(df: pd.DataFrame, delta_cols: list[str], info: dict, *, seeds: list[int], n_boot: int, fast: bool, evidence: str,
             recorder=None) -> dict:
    """`recorder(tag, seed, y, groups, nested_result)` is called after every nested fit (fold assignments / held-out predictions can be saved);
    it observes only and never changes the computation."""
    t0 = time.time()
    y, g = df["worsening"].reset_index(drop=True), df["subject#"].astype(str).reset_index(drop=True)
    thr = PROTOCOL["threshold"]
    demo = ["age", "sex"]
    sets = {"FULL": delta_cols + ["test_time", "baseline_total_UPDRS"], "VOICE_ONLY": delta_cols,
            "TIME_BASELINE": ["test_time", "baseline_total_UPDRS"], "TIME_ONLY": ["test_time"],
            "BASELINE_ONLY": ["baseline_total_UPDRS"], "DEMOGRAPHIC_ONLY": demo, "TIME_DEMOGRAPHIC": ["test_time"] + demo,
            "TIME_BASELINE_DEMOGRAPHIC": ["test_time", "baseline_total_UPDRS"] + demo}
    assert set(sets) == set(PROTOCOL["risk"]["feature_sets"]), "feature sets drifted from the frozen protocol"
    missing = [c for cols in sets.values() for c in cols if c not in df.columns]
    if missing:
        raise D.DatasetVerificationError(f"risk: columns required by the protocol are absent: {sorted(set(missing))}")
    cands = C.risk_candidates(fast)
    per_seed: dict = {s: {} for s in seeds}
    for seed in seeds:
        preds = {}
        for name, cols in sets.items():
            # exact-duplicate-row guard stays ON for the multi-column voice sets (it catches a recording duplicated in the file); it is OFF
            # for tiny baseline sets, where distinct patients legitimately share e.g. the same (age, sex). Folds do not depend on X, so the
            # FULL-set check covers the identical folds.
            res = nested_group_cv(df[cols].reset_index(drop=True), y, g, cands, outer=PROTOCOL["risk"]["outer"], inner_k=PROTOCOL["inner_k"],
                                  seed=seed, group_level_target=False, check_duplicates=name in ("FULL", "VOICE_ONLY"))
            preds[name] = res.oof
            if recorder:
                recorder(("risk", name), seed, y, g, res)
            frame = pd.DataFrame({"subject": g, "y": y, "p": res.oof})
            first = seed == seeds[0]
            per_seed[seed][name] = {
                # the full interval set is computed once (first seed); later seeds record the point estimates used for dispersion
                "pooled": full_metrics(frame, thr, n_boot, seed) if first else
                          {"roc_auc": auc_stat(frame), "pr_auc": _prauc(frame)},
                "within_patient": within_subject_auc(frame, n_boot if first else 200, seed),
                "per_outer_fold_auc": per_fold_auc(res, g, y), "chosen_counts": res.chosen_counts(),
                "n_fallback_selections": sum(f.selection_status != "selected" for f in res.folds)}
        for ref in PROTOCOL["reference_for_voice_value"]:
            fr = pd.DataFrame({"subject": g, "y": y, "p_full": preds["FULL"], "p_ref": preds[ref]})
            per_seed[seed][f"FULL_minus_{ref}"] = paired_cluster_bootstrap_diff(fr, "subject", "p_full", "p_ref", n_boot=n_boot, seed=seed)
    # dispersion across seeds (different outer fold assignments and inner splits)
    dispersion = {}
    for name in sets:
        vals = []
        for sd in seeds:
            vals.append(per_seed[sd][name]["pooled"]["roc_auc"])
        vals = [float(v) for v in vals if v is not None]
        dispersion[name] = {"auc_by_seed": vals, "mean": float(np.mean(vals)) if vals else None,
                            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                            "min": min(vals) if vals else None, "max": max(vals) if vals else None}
    voice_value = {}
    for ref in PROTOCOL["reference_for_voice_value"]:
        cis = [per_seed[s][f"FULL_minus_{ref}"] for s in seeds]
        excl = [(c.get("lo") is not None and c["lo"] > 0) for c in cis]
        voice_value[ref] = {"point_by_seed": [c["point"] for c in cis], "ci_lo_by_seed": [c.get("lo") for c in cis],
                            "ci_hi_by_seed": [c.get("hi") for c in cis], "interval_excludes_zero_by_seed": excl,
                            "meets_registered_rule": bool(all(excl))}
    rule = all(v["meets_registered_rule"] for v in voice_value.values())
    return envelope(
        "risk_concurrent_worsening_classifier", "completed", evidence, dataset=info, seeds=seeds,
        design={"outer": f"patient-grouped stratified {PROTOCOL['risk']['outer']}-fold, re-drawn per seed",
                "inner": "StratifiedGroupKFold(5) on outer-training rows only", "preprocessing": "inside each sklearn Pipeline",
                "threshold": thr, "unit_of_resampling": "patient"},
        results_by_seed=per_seed, dispersion_across_seeds=dispersion,
        voice_incremental_value={"by_reference": voice_value, "registered_rule_met": bool(rule),
                                 "reading": ("the registered rule is met: the data show a voice contribution beyond the references on THIS data set. "
                                             "That is a statement about association, not clinical utility.") if rule else
                                            ("the registered rule is NOT met: these data cannot show that voice features add information "
                                             "beyond the non-voice references.")},
        limitations=["labels derive from linearly interpolated UPDRS (publisher), so within a patient they are a smooth function of time: "
                     "effective independent information is about 42 patients x a few clinic assessments, not thousands of recordings",
                     "target is concurrent (is this visit worse than the first?), not a forecast",
                     "one telemonitoring device, one study; no external cohort",
                     "model selection among 4 candidate configurations is nested; the final saved model is not what is scored",
                     "scores are not calibrated probabilities", "clinically_validated is false whatever the numbers"],
        runtime_seconds=round(time.time() - t0, 1))


def run_disease(df: pd.DataFrame, info: dict, *, seeds: list[int], n_boot: int, n_perm: int, fast: bool, evidence: str,
                recorder=None) -> dict:
    t0 = time.time()
    X = df[D.DISEASE_FEATURES].reset_index(drop=True)
    y, g = df["status"].reset_index(drop=True), df["subject"].reset_index(drop=True)
    thr = PROTOCOL["threshold"]
    cands, bases = C.disease_candidates(fast), C.disease_baselines(fast)
    per_seed: dict = {}
    for seed in seeds:
        nested = nested_group_cv(X, y, g, cands, outer="logo", inner_k=PROTOCOL["inner_k"], seed=seed)
        if recorder:
            recorder(("disease", "candidates"), seed, y, g, nested)
        rec = pd.DataFrame({"subject": g, "y": y, "p": nested.oof})
        subj = aggregate_by_group(g, y, nested.oof, "mean").rename(columns={"group": "subject"})
        entry = {"recording_level": full_metrics(rec, thr, n_boot, seed) if seed == seeds[0] else evaluate_binary(rec["y"], rec["p"], thr),
                 "subject_level": {**evaluate_binary(subj["y"], subj["p"], thr),
                                   "roc_auc_cluster_bootstrap_ci95": cluster_bootstrap(subj, "subject", auc_stat, n_boot=n_boot, seed=seed)},
                 "chosen_counts": nested.chosen_counts(),
                 "n_fallback_selections": sum(f.selection_status != "selected" for f in nested.folds)}
        sub_pred = subj.assign(pred=(subj["p"] >= thr).astype(int))
        entry["subject_level"]["healthy_subjects_correct"] = int(((sub_pred["y"] == 0) & (sub_pred["pred"] == 0)).sum())
        entry["subject_level"]["healthy_subjects_total"] = int((sub_pred["y"] == 0).sum())
        entry["subject_level"]["pd_subjects_correct"] = int(((sub_pred["y"] == 1) & (sub_pred["pred"] == 1)).sum())
        entry["subject_level"]["pd_subjects_total"] = int((sub_pred["y"] == 1).sum())
        k0, n0 = entry["subject_level"]["healthy_subjects_correct"], entry["subject_level"]["healthy_subjects_total"]
        k1, n1 = entry["subject_level"]["pd_subjects_correct"], entry["subject_level"]["pd_subjects_total"]
        entry["subject_level"]["specificity_clopper_pearson95"] = clopper_pearson(k0, n0)
        entry["subject_level"]["sensitivity_clopper_pearson95"] = clopper_pearson(k1, n1)
        entry["baselines"] = {}
        base_oof = {}
        for bname, fac in bases.items():
            r = nested_group_cv(X, y, g, {bname: fac}, outer="logo", inner_k=PROTOCOL["inner_k"], seed=seed)
            base_oof[bname] = r.oof
            if recorder:
                recorder(("disease", bname), seed, y, g, r)
            s2 = aggregate_by_group(g, y, r.oof, "mean").rename(columns={"group": "subject"})
            rr = pd.DataFrame({"subject": g, "y": y, "p": r.oof})
            ev = noskill_metrics if bname == "dummy_prior" else evaluate_binary
            entry["baselines"][bname] = {"recording_level": ev(rr["y"], rr["p"], thr),
                                         "subject_level": ev(s2["y"], s2["p"], thr)}
        pair = subj.assign(p_base=aggregate_by_group(g, y, base_oof["classic3_LR"], "mean")["p"].to_numpy())
        entry["paired_auc_difference_vs_classic3_LR_subject_level"] = paired_cluster_bootstrap_diff(pair, "subject", "p", "p_base", n_boot=n_boot, seed=seed)
        entry["per_subject"] = [{"subject": r.subject, "label": int(r.y), "mean_score": float(r.p)} for r in subj.itertuples()]
        if seed == seeds[0]:
            modal = max(nested.chosen_counts(), key=nested.chosen_counts().get)

            def p_fn(yp, _m=modal):
                oof = np.zeros(len(yp))
                for tr, te in leave_one_group_out(g, X=X):
                    m = cands[_m]()
                    m.fit(X.iloc[tr], yp[tr])
                    oof[te] = m.predict_proba(X.iloc[te])[:, 1]
                return oof

            entry["permutation_test_fixed_pipeline"] = {
                "pipeline": modal, **group_label_permutation_test(g, y.to_numpy(), p_fn, auc_stat(rec), n_perm=n_perm, seed=seed),
                "note": "tests association of a FIXED pipeline (the modal nested choice) with the label under subject-level label "
                        "permutation; it does not test the selection procedure and does not show clinical utility"}
        per_seed[seed] = entry
    aucs = [per_seed[s]["subject_level"]["roc_auc"] for s in seeds if per_seed[s]["subject_level"]["roc_auc"] is not None]
    return envelope(
        "disease_voice_classifier", "completed", evidence, dataset=info, seeds=seeds,
        design={"outer": "leave-one-subject-out", "inner": "StratifiedGroupKFold(5) on outer-training rows only",
                "preprocessing": "inside each sklearn Pipeline (scaler, feature selection)", "threshold": thr, "unit_of_resampling": "subject"},
        results_by_seed=per_seed,
        dispersion_across_seeds={"subject_level_auc_by_seed": aucs, "sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
                                 "min": min(aucs) if aucs else None, "max": max(aucs) if aucs else None},
        limitations=["about 31-32 subjects of which 8 healthy: every healthy subject moves specificity by 12.5 points",
                     "single recording protocol from one lab; recordings of one person are highly similar; no external cohort",
                     "study prevalence (~75% PD) is not a screening population: see the PPV/NPV table in docs/phase4",
                     "pooled leave-one-subject-out AUC is biased below 0.5 under the null (training prior shifts against the held-out "
                     "subject); the permutation test uses the same pooled statistic so it is unaffected, the raw number is not comparable to 0.5",
                     "clinically_validated is false whatever the numbers"],
        ppv_npv_at_prevalence={str(p): ppv_npv_at_prevalence(per_seed[seeds[0]]["subject_level"]["at_threshold"]["sensitivity"],
                                                             per_seed[seeds[0]]["subject_level"]["at_threshold"]["specificity"], p)
                               for p in (0.005, 0.01, 0.05, 0.2, 0.75)}
        if per_seed[seeds[0]]["subject_level"]["at_threshold"]["sensitivity"] is not None
        and per_seed[seeds[0]]["subject_level"]["at_threshold"]["specificity"] is not None else None,
        runtime_seconds=round(time.time() - t0, 1))


# ------------------------------------------------------------------------------------------------ synthetic self-test
def synthetic_disease(seed: int, n_subjects: int = 20, rec: int = 5, signal: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """SYNTHETIC. Strong between-subject random effects (so a row-level split would leak) and group-level labels.
    signal=0 -> labels carry no information about features."""
    rng = np.random.default_rng(seed)
    rows = []
    n_pd = round(n_subjects * 0.75)
    for s in range(1, n_subjects + 1):
        lab = int(s <= n_pd)
        eff = rng.normal(0, 2.0, size=len(D.DISEASE_FEATURES))           # subject fingerprint
        for r in range(1, rec + 1):
            vals = eff + rng.normal(0, 0.3, size=len(D.DISEASE_FEATURES))
            vals[:3] += signal * lab
            rows.append({"name": f"phon_R01_S{s:02d}_{r}", "status": lab, **dict(zip(D.DISEASE_FEATURES, vals.tolist()))})
    df = pd.DataFrame(rows)
    info = D.verify_disease_frame(df, strict_counts=False)
    df["subject"] = df["name"].str.extract(r"(S\d+)_\d+$")[0]
    return df, {**info, "SYNTHETIC": True}


def synthetic_risk(seed: int, n_patients: int = 16, visits: int = 30, voice_signal: float = 0.0):
    """SYNTHETIC. Labels are driven by time and baseline severity (as in the real task's construction); voice deltas carry signal only
    if voice_signal > 0."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_patients + 1):
        base, slope = rng.uniform(15, 40), rng.normal(0.03, 0.03)
        age, sex = int(rng.integers(50, 80)), int(rng.integers(0, 2))
        eff = rng.normal(0, 1, size=3)
        for t in np.sort(rng.uniform(0, 180, visits)):
            upd = base + slope * t + rng.normal(0, 0.8)
            v = eff + rng.normal(0, 1, size=3) + voice_signal * (upd - base) / 5.0
            rows.append({"subject#": s, "age": age, "sex": sex, "test_time": float(t), "motor_UPDRS": 0.0, "total_UPDRS": upd,
                         "Jitter(%)": v[0], "Shimmer": v[1], "HNR": v[2]})
    raw = pd.DataFrame(rows)
    info = D.verify_risk_frame(raw, strict_counts=False)
    df, dcols = D.build_risk_frame(raw)
    info.update({"n_rows": len(df), "positive_rate": float(df["worsening"].mean()), "SYNTHETIC": True,
                 "worsening_threshold_points": D.WORSENING_THRESHOLD})
    return df, dcols, info


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["artifacts", "authenticate", "run", "selftest"])
    ap.add_argument("--kind", choices=["disease", "risk"])
    ap.add_argument("--csv")
    ap.add_argument("--seeds", type=int, default=PROTOCOL["default_seeds"])
    ap.add_argument("--n-boot", type=int, default=PROTOCOL["n_boot"])
    ap.add_argument("--n-perm", type=int, default=PROTOCOL["n_perm"])
    ap.add_argument("--fast", action="store_true", help="fewer trees; for smoke tests only, NOT the registered protocol")
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    seeds = [PROTOCOL["seed_base"] + i for i in range(a.seeds)]
    try:
        if a.cmd == "artifacts":
            print(write("artifact_audit", envelope("artifact_audit", "completed", "REAL_ARTIFACTS_NO_DATA_USED", **artifact_audit()), out_dir))
            return 0
        if a.cmd == "selftest":
            from ml.validation.phase4a_selftest import run_selftest
            print(write("selftest_SYNTHETIC_NOT_EVIDENCE", run_selftest(), out_dir))
            return 0
        if not a.kind:
            ap.error("--kind is required")
        try:
            path = resolve_data_path(a.kind, a.csv)
        except DataUnavailable as e:
            print(write(f"{a.kind}_NOT_RUN", envelope(a.kind, "not_run", "NONE", reason=str(e)), out_dir), file=sys.stderr)
            print(f"NOT RUN: {e}", file=sys.stderr)
            return 2
        auth = F.authenticate(a.kind, path)
        if a.cmd == "authenticate":
            print(write(f"{a.kind}_authentication", envelope(a.kind, "completed", F.evidence_status(auth), authentication=auth), out_dir))
            return 0 if auth["status"] == F.AUTHENTICATED else 3
        if a.fast:
            print("WARNING: --fast is not the registered protocol; the output is marked accordingly.", file=sys.stderr)
        evidence = F.evidence_status(auth) + ("+FAST_NOT_REGISTERED_PROTOCOL" if a.fast else "")
        if a.kind == "disease":
            df, info = D.load_disease(path)
            res = run_disease(df, {**info, "path": str(path)}, seeds=seeds, n_boot=a.n_boot, n_perm=a.n_perm, fast=a.fast, evidence=evidence)
        else:
            df, dcols, info = D.load_risk(path)
            res = run_risk(df, dcols, {**info, "path": str(path)}, seeds=seeds, n_boot=a.n_boot, fast=a.fast, evidence=evidence)
        print(write(f"{a.kind}_p4a", {**res, "authentication": auth}, out_dir))
        return 0
    except Exception as e:  # a failed run is REPORTED as failed; there is no partial success
        payload = envelope(a.cmd, "failed", "NONE", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        print(write(f"{a.kind or a.cmd}_FAILED", payload, out_dir), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
