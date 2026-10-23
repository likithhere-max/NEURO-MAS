"""
Phase 6 report: comprehensive metrics, paired comparisons, calibration, decision curves, subgroups and error analysis, computed
from the HELD-OUT PREDICTIONS saved by a workflow run (`ml.validation.workflow`).  Nothing is refitted here.

    PYTHONPATH=. python -m ml.validation.phase6_report --p4a-dir RUN --out OUT [--ext-dir EXT_RUN] [--risk-csv parkinsons_updrs.data]

Gates (all fail closed; a gate failure writes `phase6_NOT_RUN.json`, exits 2 and prints no numbers):
  * the run's experiment_manifest.json must exist and the experiment must be `completed`;
  * the evidence label is COPIED from the run (never upgraded); a synthetic or unauthenticated run keeps its label;
  * extension predictions are compared with p4a-1 predictions only when the outer-fold assignment is identical row by row;
  * covariates for subgroups are used only if the supplied file reproduces the saved labels row by row.

Output: `phase6_metrics.json` (machine-readable), `MODEL_COMPARISON.generated.md`, `ERROR_ANALYSIS.generated.md`.  Every number in the
markdown is registered against the JSON and re-verified (ml.validation.report_from_artifacts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evalkit import clinical_metrics as CM
from ml.validation import data as D
from ml.validation import phase4a as P
from ml.validation import phase6 as PH6
from ml.validation import report_from_artifacts as R

REPORT_VERSION = "1.0.0"
THRESHOLD = PH6.PROTOCOL_P6["threshold"]
DCA_GRID = PH6.PROTOCOL_P6["decision_curve"]["threshold_grid"]
RISK_P4A_SETS = tuple(P.PROTOCOL["risk"]["feature_sets"])
VOICE_PAIRS = (("FULL", "TIME_BASELINE"), ("FULL", "TIME_BASELINE_DEMOGRAPHIC"), ("TIME_BASELINE", "TIME_ONLY"), ("FULL", "VOICE_ONLY"))
PAIR_METRICS = ("roc_auc", "pr_auc", "brier", "log_loss")


class Gate(RuntimeError):
    pass


# ----------------------------------------------------------------------------- loading and gating
def _load_manifest(run_dir: Path) -> dict:
    p = Path(run_dir) / "experiment_manifest.json"
    if not p.exists():
        raise Gate(f"{run_dir}: no experiment_manifest.json (run `ml.validation.workflow run` first)")
    return json.loads(p.read_text())


def _gate(run_dir: Path, kind: str, expect_candidate_set: str) -> tuple[dict, str]:
    m = _load_manifest(run_dir)
    if m.get("candidate_set", "p4a-1") != expect_candidate_set:
        raise Gate(f"{run_dir} holds candidate set {m.get('candidate_set')!r}, expected {expect_candidate_set!r}")
    e = m["experiments"].get(kind)
    if not e or e["status"] != "completed":
        raise Gate(f"{kind}: experiment status is {e['status'] if e else 'absent'} ({(e or {}).get('reason', 'no record')})")
    return m, e["evidence_status"]


def _predictions(run_dir: Path, kind: str, prefix: str) -> pd.DataFrame:
    p = Path(run_dir) / "protected" / f"{prefix}{kind}_heldout_predictions.csv"
    if not p.exists():
        raise Gate(f"{kind}: held-out prediction file is missing ({p.name}); metrics cannot be recomputed")
    df = pd.read_csv(p, dtype={"subject": str})
    need = {"kind", "model_tag", "seed", "row", "subject", "y", "p", "outer_fold"}
    if not need <= set(df.columns):
        raise Gate(f"{kind}: prediction file lacks columns {sorted(need - set(df.columns))}")
    if df[["y", "p", "outer_fold"]].isna().any().any():
        raise Gate(f"{kind}: prediction file has missing y/p/outer_fold values (a row was never held out)")
    if not df["p"].between(0, 1).all():
        raise Gate(f"{kind}: scores outside [0, 1]")
    return df


def _slice(df: pd.DataFrame, tag: str, seed: int) -> pd.DataFrame:
    s = df[(df["model_tag"] == tag) & (df["seed"] == seed)].sort_values("row").reset_index(drop=True)
    if s.empty:
        raise Gate(f"no predictions for {tag!r} at seed {seed}")
    if s["row"].tolist() != list(range(len(s))):
        raise Gate(f"{tag!r} seed {seed}: row index is not a complete 0..n-1 sequence")
    return s


def _same_folds(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return (len(a) == len(b) and (a["subject"].to_numpy() == b["subject"].to_numpy()).all() and (a["y"].to_numpy() == b["y"].to_numpy()).all()
            and (a["outer_fold"].to_numpy() == b["outer_fold"].to_numpy()).all())


# ----------------------------------------------------------------------------- analyses
def _model_block(s: pd.DataFrame, n_boot: int, seed: int) -> dict:
    c = CM.comprehensive(s["y"].to_numpy(), s["p"].to_numpy(), s["subject"].to_numpy(), threshold=THRESHOLD, n_boot=n_boot, seed=seed)
    c["decision_curve"] = CM.net_benefit(s["y"], s["p"], DCA_GRID)
    c["reliability"] = P.reliability_table(s["y"], s["p"], 5)
    return c


def _pair_block(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    return CM.paired_differences(a["y"].to_numpy(), a["p"].to_numpy(), b["p"].to_numpy(), a["subject"].to_numpy(), metrics=PAIR_METRICS,
                                 threshold=THRESHOLD, n_boot=n_boot, seed=seed)


def analyse_risk(p4a_dir: Path, ext_dir: Path | None, risk_csv: Path | None, n_boot: int) -> dict:
    m, evidence = _gate(p4a_dir, "risk", "p4a-1")
    preds = _predictions(p4a_dir, "risk", "")
    seeds = sorted(preds["seed"].unique().tolist())
    first = seeds[0]
    tags = [t for t in RISK_P4A_SETS if t in set(preds["model_tag"])]
    out: dict = {"status": "completed", "evidence_status": evidence, "seeds": seeds, "threshold": THRESHOLD, "n_boot": n_boot, "models": {}, "by_seed_point": {},
                 "paired": {}, "voice_value": {}, "extension": None, "subgroups": None}
    for t in tags:
        out["models"][t] = _model_block(_slice(preds, t, first), n_boot, first)
        out["by_seed_point"][t] = {str(sd): CM.point_metrics(*(lambda s: (s["y"], s["p"]))(_slice(preds, t, sd)), THRESHOLD) for sd in seeds}
    # paired comparisons on every seed (the registered voice rule needs every seed)
    for a, b in VOICE_PAIRS:
        if a in tags and b in tags:
            out["paired"][f"{a}_minus_{b}"] = {str(sd): _pair_block(_slice(preds, a, sd), _slice(preds, b, sd), n_boot, sd) for sd in seeds}
    refs = [r for r in P.PROTOCOL["reference_for_voice_value"] if f"FULL_minus_{r}" in out["paired"]]
    if refs:
        by_ref = {}
        for r in refs:
            iv = [out["paired"][f"FULL_minus_{r}"][str(sd)]["roc_auc"] for sd in seeds]
            excl = [x["lo"] is not None and x["lo"] > 0 for x in iv]
            by_ref[r] = {"point_by_seed": [x["point_difference_a_minus_b"] for x in iv], "lo_by_seed": [x["lo"] for x in iv],
                         "hi_by_seed": [x["hi"] for x in iv], "excludes_zero_by_seed": excl, "meets_rule": bool(all(excl))}
        met = all(v["meets_rule"] for v in by_ref.values()) and len(refs) == len(P.PROTOCOL["reference_for_voice_value"])
        out["voice_value"] = {"by_reference": by_ref, "rule_met_recomputed": bool(met),
                              "verdict": "VOICE_ADDS_MEASURABLE_INFORMATION_ON_THIS_DATA" if met else "INCONCLUSIVE",
                              "reading": ("The registered rule is met on these data (association only; not clinical utility)." if met else
                                          "The registered rule is not met: these data cannot show that voice adds information. This is NOT evidence that it adds none; "
                                          "see the sample-size planning for what a conclusive test needs."),
                              "p4a_recorded_verdict": _recorded_verdict(p4a_dir)}
    # exploratory recalibration of the best-ranking registered model is NOT chosen from outcomes: it is applied to FULL and TIME_BASELINE as fixed
    out["recalibration_exploratory"] = {}
    for t in ("FULL", "TIME_BASELINE"):
        if t in tags:
            s = _slice(preds, t, first)
            rc = CM.cross_fitted_recalibration(s["y"], s["p"], s["subject"], k=5, seed=first)
            rc.pop("recalibrated", None)
            out["recalibration_exploratory"][t] = rc
    if ext_dir:
        out["extension"] = _extension(ext_dir, preds, tags, seeds, n_boot)
    out["subgroups"] = _subgroups(preds, tags, first, risk_csv, n_boot)
    out["error_analysis"] = _risk_errors(preds, tags, first, risk_csv)
    return out


def _recorded_verdict(p4a_dir: Path):
    p = Path(p4a_dir) / "risk_p4a.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text()).get("voice_incremental_value", {})
    return {"registered_rule_met": v.get("registered_rule_met"), "source": "risk_p4a.json (the registered p4a-1 computation; authoritative)"}


def _extension(ext_dir: Path, base: pd.DataFrame, tags: list[str], seeds: list[int], n_boot: int) -> dict:
    try:
        _, ev = _gate(ext_dir, "risk", PH6.EXT_ID)
        ext = _predictions(ext_dir, "risk", "p6ext_")
    except Gate as e:
        return {"status": "not_run", "reason": str(e)}
    first = seeds[0]
    res: dict = {"status": "completed", "evidence_status": ev, "label": "EXPLORATORY", "sets": {}}
    for t in PH6.RISK_SETS:
        et = f"{t}|ext"
        if t not in tags or et not in set(ext["model_tag"]):
            continue
        a, b = _slice(ext, et, first), _slice(base, t, first)
        if not _same_folds(a, b):
            res["sets"][t] = {"status": "refused", "reason": "outer-fold assignment differs between the extension and p4a-1 runs: paired comparison would be invalid"}
            continue
        res["sets"][t] = {"status": "completed", "extension": _model_block(a, n_boot, first), "paired_extension_minus_p4a1": _pair_block(a, b, n_boot, first)}
    return res


def _risk_frame(risk_csv: Path | None, preds: pd.DataFrame, tag: str, seed: int):
    if risk_csv is None:
        return None, "covariates not supplied (--risk-csv)"
    df, _, _ = D.load_risk(Path(risk_csv))
    df = df.reset_index(drop=True)
    s = _slice(preds, tag, seed)
    if len(df) != len(s) or (df["subject#"].astype(str).to_numpy() != s["subject"].to_numpy()).any() or (df["worsening"].to_numpy() != s["y"].to_numpy()).any():
        return None, "the supplied file does not reproduce the saved subjects and labels row by row: subgroups not computed"
    return df, None


def _tertile(v: pd.Series, by_subject: pd.Series) -> pd.Series:
    cuts = np.quantile(by_subject.to_numpy(), [1 / 3, 2 / 3])        # cut points from one value per subject (a covariate), never from outcomes
    return pd.cut(v, [-np.inf, cuts[0], cuts[1], np.inf], labels=["T1_low", "T2_mid", "T3_high"], duplicates="drop")


def _subgroups(preds: pd.DataFrame, tags: list[str], seed: int, risk_csv, n_boot: int) -> dict:
    tag = "FULL" if "FULL" in tags else (tags[0] if tags else None)
    if tag is None:
        return {"status": "not_run", "reason": "no model predictions"}
    df, why = _risk_frame(risk_csv, preds, tag, seed)
    if df is None:
        return {"status": "not_run", "reason": why}
    out = {"status": "completed", "model": tag, "label": "DESCRIPTIVE; levels below the fixed minimum size are listed as insufficient", "groups": {}}
    subj_first = df.groupby("subject#").first()
    cov = pd.DataFrame({"sex": df["sex"].map({0: "sex=0", 1: "sex=1"}),
                        "age_tertile": _tertile(df["age"], subj_first["age"]).astype(str),
                        "baseline_severity_tertile": _tertile(df["baseline_total_UPDRS"], subj_first["baseline_total_UPDRS"]).astype(str)})
    for tg in {"FULL", "TIME_BASELINE"} & set(tags):
        s = _slice(preds, tg, seed)
        for col in cov.columns:
            frame = s[["y", "p", "subject"]].assign(**{col: cov[col].to_numpy()})
            out["groups"][f"{tg}|{col}"] = CM.subgroup_report(frame, col, threshold=THRESHOLD, n_boot=max(200, n_boot // 4), seed=seed)
    return out


def _risk_errors(preds: pd.DataFrame, tags: list[str], seed: int, risk_csv) -> dict:
    tag = "FULL" if "FULL" in tags else (tags[0] if tags else None)
    if tag is None:
        return {"status": "not_run"}
    s = _slice(preds, tag, seed)
    s = s.assign(err=(s["p"] - s["y"]).abs(), wrong=((s["p"] >= THRESHOLD).astype(int) != s["y"]).astype(int))
    per = s.groupby("subject").agg(n=("y", "size"), prevalence=("y", "mean"), accuracy=("wrong", lambda w: 1 - w.mean()), brier=("err", lambda e: float((e ** 2).mean())))
    per["auc"] = [CM.point_metrics(g["y"], g["p"], THRESHOLD)["roc_auc"] for _, g in s.groupby("subject")]
    worst = per.sort_values("brier", ascending=False).head(5).reset_index()
    out = {"status": "completed", "model": tag, "n_subjects": int(len(per)),
           "per_subject_accuracy": {"min": float(per["accuracy"].min()), "median": float(per["accuracy"].median()), "max": float(per["accuracy"].max())},
           "subjects_with_defined_auc": int(per["auc"].notna().sum()),
           "per_subject_auc_median": None if per["auc"].notna().sum() == 0 else float(per["auc"].median()),
           "per_subject_auc_below_half": int((per["auc"] < 0.5).sum()),
           "five_highest_brier_subjects": worst.to_dict("records"),
           "per_outer_fold_auc": [{"fold": int(f), **{"auc": CM.point_metrics(g["y"], g["p"], THRESHOLD)["roc_auc"], "n": int(len(g))}} for f, g in s.groupby("outer_fold")]}
    df, why = _risk_frame(risk_csv, preds, tag, seed)
    if df is None:
        out["by_time_note"] = why
    else:
        q = pd.qcut(df["test_time"], 4, duplicates="drop")
        out["error_rate_by_test_time_quartile"] = [{"bin": str(k), "n": int(len(g)), "error_rate": float(s.loc[g.index, "wrong"].mean())} for k, g in df.groupby(q, observed=True)]
    return out


def analyse_disease(p4a_dir: Path, ext_dir: Path | None, n_boot: int) -> dict:
    m, evidence = _gate(p4a_dir, "disease", "p4a-1")
    preds = _predictions(p4a_dir, "disease", "")
    seeds = sorted(preds["seed"].unique().tolist())
    first = seeds[0]
    out: dict = {"status": "completed", "evidence_status": evidence, "seeds": seeds, "threshold": THRESHOLD, "n_boot": n_boot, "models": {}, "paired": {}, "extension": None}
    for tag in sorted(set(preds["model_tag"])):
        s = _slice(preds, tag, first)
        subj = s.groupby("subject").agg(y=("y", "first"), p=("p", "mean")).reset_index()
        blk = {"recording_level": _model_block(s, n_boot, first),
               "subject_level": {"n_subjects": int(len(subj)), "n_pd": int(subj["y"].sum()), "n_healthy": int((subj["y"] == 0).sum()),
                                 "note": "one mean score per subject; the unit of inference is the subject; intervals are wide with so few healthy subjects",
                                 "result": CM.comprehensive(subj["y"].to_numpy(), subj["p"].to_numpy(), subj["subject"].to_numpy(), threshold=THRESHOLD, n_boot=n_boot, seed=first)}}
        out["models"][tag] = blk
    if {"candidates", "classic3_LR"} <= set(out["models"]):
        a, b = _slice(preds, "candidates", first), _slice(preds, "classic3_LR", first)
        out["paired"]["candidates_minus_classic3_LR"] = _pair_block(a, b, n_boot, first)
    s = _slice(preds, "candidates", first) if "candidates" in set(preds["model_tag"]) else None
    if s is not None:
        subj = s.groupby("subject").agg(y=("y", "first"), p=("p", "mean"), n=("y", "size")).reset_index()
        wrong = subj[(subj["p"] >= THRESHOLD).astype(int) != subj["y"]]
        out["error_analysis"] = {"model": "candidates (nested winner per fold)", "n_subjects": int(len(subj)), "n_misclassified_subjects": int(len(wrong)),
                                 "misclassified_healthy": wrong[wrong["y"] == 0][["subject", "p", "n"]].to_dict("records"),
                                 "misclassified_pd": wrong[wrong["y"] == 1][["subject", "p", "n"]].to_dict("records")}
    if ext_dir:
        try:
            _, ev = _gate(ext_dir, "disease", PH6.EXT_ID)
            ext = _predictions(ext_dir, "disease", "p6ext_")
            a, b = _slice(ext, "ext", first), _slice(preds, "candidates", first)
            out["extension"] = ({"status": "completed", "evidence_status": ev, "label": "EXPLORATORY", "extension": _model_block(a, n_boot, first),
                                 "paired_extension_minus_p4a1": _pair_block(a, b, n_boot, first)}
                                if _same_folds(a, b) else {"status": "refused", "reason": "outer-fold assignment differs: paired comparison invalid"})
        except Gate as e:
            out["extension"] = {"status": "not_run", "reason": str(e)}
    return out


# ----------------------------------------------------------------------------- rendering (every number registered and verified)
def _fmt3(b: R._Builder, path: str) -> str:
    return b.num("phase6_metrics.json", path, "3")


def _metric_row(b: R._Builder, label: str, base: str) -> str:
    cells = []
    for k in ("roc_auc", "pr_auc", "brier", "log_loss", "sensitivity", "specificity", "calibration_slope"):
        pt = f"{base}.metrics.{k}.point"
        lo, hi = f"{base}.metrics.{k}.lo", f"{base}.metrics.{k}.hi"
        cells.append(f"{_fmt3(b, pt)} [{_fmt3(b, lo)}, {_fmt3(b, hi)}]")
    return f"| {label} | " + " | ".join(cells) + " |"


def render_comparison(results_dir: Path) -> tuple[str, list[dict]]:
    b = R._Builder(results_dir)
    j = b.load("phase6_metrics.json")
    b.add(R.BEGIN)
    for kind, title in (("risk", "Risk classifier (concurrent worsening)"), ("disease", "Disease classifier (PD vs healthy)")):
        blk = (j or {}).get(kind)
        b.add(f"### {title}")
        if not blk or blk.get("status") != "completed":
            b.add(f"Status: **NOT RUN** - {(blk or {}).get('reason', 'no result')}. No results are shown.")
            b.add("")
            continue
        b.add(f"Status: **completed**. Evidence label: `{blk['evidence_status']}`. Threshold fixed at 0.5. Cluster-bootstrap 95% intervals "
              "(resampling subjects). First seed shown; other seeds in the JSON. `clinically_validated` = False.")
        b.add("")
        b.add("| Model | ROC-AUC | PR-AUC | Brier | Log loss | Sensitivity | Specificity | Calibration slope |")
        b.add("|---|---|---|---|---|---|---|---|")
        if kind == "risk":
            for t in blk["models"]:
                b.add(_metric_row(b, t, f"{kind}.models.{t}"))
            ext = blk.get("extension") or {}
            if ext.get("status") == "completed":
                b.add("")
                b.add(f"Extension families (`{PH6.EXT_ID}`, **EXPLORATORY**, evidence label `{ext['evidence_status']}`), same folds as the rows above:")
                b.add("")
                b.add("| Model | ROC-AUC | PR-AUC | Brier | Log loss | Sensitivity | Specificity | Calibration slope |")
                b.add("|---|---|---|---|---|---|---|---|")
                for t, blk_t in ext["sets"].items():
                    if blk_t.get("status") == "completed":
                        b.add(_metric_row(b, f"{t} (extension)", f"{kind}.extension.sets.{t}.extension"))
                    else:
                        b.add(f"| {t} (extension) | refused: {blk_t.get('reason')} |  |  |  |  |  |  |")
            vv = blk.get("voice_value") or {}
            if vv:
                b.add("")
                b.add(f"Registered voice-value rule recomputed from the saved predictions: **{vv['verdict']}**.")
        else:
            for t in blk["models"]:
                b.add(_metric_row(b, f"{t} (recordings)", f"{kind}.models.{t}.recording_level"))
                b.add(_metric_row(b, f"{t} (subjects)", f"{kind}.models.{t}.subject_level.result"))
        b.add("")
    b.add(R.END)
    return "\n".join(b.lines) + "\n", b.claims


def render_errors(results_dir: Path) -> tuple[str, list[dict]]:
    b = R._Builder(results_dir)
    j = b.load("phase6_metrics.json")
    b.add(R.BEGIN)
    r = ((j or {}).get("risk") or {}).get("error_analysis")
    b.add("### Risk classifier errors")
    if not r or r.get("status") != "completed":
        b.add("Status: **NOT RUN**. No results are shown.")
    else:
        b.add(f"Status: **completed**. Model `{r['model']}`, first seed.")
        b.add("")
        b.add("| Statistic | Value |")
        b.add("|---|---|")
        for lab, path in (("Subjects", "n_subjects"), ("Per-subject accuracy, minimum", "per_subject_accuracy.min"),
                          ("Per-subject accuracy, median", "per_subject_accuracy.median"), ("Per-subject accuracy, maximum", "per_subject_accuracy.max"),
                          ("Subjects with per-subject AUC below 0.5", "per_subject_auc_below_half")):
            fmt = "d" if path in ("n_subjects", "per_subject_auc_below_half") else "3"
            b.add(f"| {lab} | {b.num('phase6_metrics.json', 'risk.error_analysis.' + path, fmt)} |")
    b.add("")
    d = ((j or {}).get("disease") or {}).get("error_analysis")
    b.add("### Disease classifier errors")
    if not d:
        b.add("Status: **NOT RUN**. No results are shown.")
    else:
        b.add(f"Status: **completed**. Model `{d['model']}`, first seed.")
        b.add("")
        b.add("| Statistic | Value |")
        b.add("|---|---|")
        b.add(f"| Subjects | {b.num('phase6_metrics.json', 'disease.error_analysis.n_subjects', 'd')} |")
        b.add(f"| Misclassified subjects | {b.num('phase6_metrics.json', 'disease.error_analysis.n_misclassified_subjects', 'd')} |")
    b.add("")
    b.add(R.END)
    return "\n".join(b.lines) + "\n", b.claims


def _write_md(results_dir: Path, name: str, title: str, md: str, claims: list[dict]) -> None:
    R.verify(md, claims, results_dir)
    (results_dir / name).write_text(f"# {title}\n\nEvery number below was read from `phase6_metrics.json` and re-verified against it.\n\n{md}")
    (results_dir / name.replace(".md", ".claims.json")).write_text(json.dumps(claims, indent=1) + "\n")


# ----------------------------------------------------------------------------- driver
def build(p4a_dir: Path, out_dir: Path, *, ext_dir: Path | None = None, risk_csv: Path | None = None, n_boot: int = 1000) -> tuple[int, dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("phase6_metrics.json", "phase6_NOT_RUN.json", "MODEL_COMPARISON.generated.md", "MODEL_COMPARISON.generated.claims.json",
                  "ERROR_ANALYSIS.generated.md", "ERROR_ANALYSIS.generated.claims.json"):
        (out / stale).unlink(missing_ok=True)
    result: dict = {"task": "phase6_report", "report_version": REPORT_VERSION, "status": "completed", "clinically_validated": False,
                    "protocol_id": PH6.PROTOCOL_P6["id"], "protocol_sha256": PH6.PROTOCOL_P6_SHA256,
                    "parent_protocol_id": P.PROTOCOL["id"], "parent_protocol_sha256": P.PROTOCOL_SHA256, "environment": P.env_info(),
                    "n_boot": n_boot, "inputs": {"p4a_dir": str(p4a_dir), "ext_dir": None if ext_dir is None else str(ext_dir)}}
    reasons = {}
    for kind, fn in (("risk", lambda: analyse_risk(Path(p4a_dir), ext_dir, risk_csv, n_boot)), ("disease", lambda: analyse_disease(Path(p4a_dir), ext_dir, n_boot))):
        try:
            result[kind] = fn()
        except Gate as e:
            result[kind] = {"status": "not_run", "reason": str(e)}
            reasons[kind] = str(e)
    if len(reasons) == 2:
        result["status"] = "not_run"
        (out / "phase6_NOT_RUN.json").write_text(json.dumps({**{k: v for k, v in result.items() if k not in ("risk", "disease")}, "reasons": reasons}, indent=2) + "\n")
        return 2, result
    result["overall"] = "completed" if not reasons else "INCOMPLETE"
    (out / "phase6_metrics.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    md, cl = render_comparison(out)
    _write_md(out, "MODEL_COMPARISON.generated.md", "Generated model comparison", md, cl)
    md, cl = render_errors(out)
    _write_md(out, "ERROR_ANALYSIS.generated.md", "Generated error analysis", md, cl)
    return (0 if not reasons else 2), result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p4a-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext-dir")
    ap.add_argument("--risk-csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    code, res = build(Path(a.p4a_dir), Path(a.out), ext_dir=Path(a.ext_dir) if a.ext_dir else None, risk_csv=Path(a.risk_csv) if a.risk_csv else None, n_boot=a.n_boot)
    for k in ("risk", "disease"):
        print(f"{k:8s} {res[k]['status'].upper():10s} {res[k].get('reason') or ''}", file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
