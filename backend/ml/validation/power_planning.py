"""
Design-stage power and precision planning (Phase 5, Stage 3).  evidence_status = DESIGN_STAGE_PLANNING_NOT_EVIDENCE.

None of this is a result about the Parkinson's models.  It answers: "given how few independent units the data have, what could a
study of this shape possibly detect?"  Three parts:

  analytic_disease_precision()   closed-form Hanley-McNeil standard error of a subject-level AUC for n_pd PD and n_h healthy SUBJECTS,
                                 and the minimum AUC (vs 0.5) detectable at 80% power.  Assumes one independent score per subject, fixed
                                 scores (ignores training variability) -> OPTIMISTIC.
  prevalence_sensitivity()       PPV/NPV of the shipped models' HISTORICAL (unverified) sensitivity/specificity across prevalences: arithmetic.
  simulate_risk_voice_power()    the registered voice-value rule (paired AUC difference, patient-cluster bootstrap) on synthetic data shaped like
                                 the real design (42 patients, ~139 sessions each); the 'voice signal' unit is the generator's own, so the
                                 transferable output is the AUC DIFFERENCE at which detection reaches a given rate, not the signal value.

Run:  PYTHONPATH=. python -m ml.validation.power_planning [--reps 20] [--out ml/validation/results/phase5/power_planning.json]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ml.validation import phase4a as P
from ml.validation import phase4a_selftest as ST

Z_ALPHA, Z_POWER = 1.959964, 0.841621             # two-sided 5%, 80% power
OUT = Path(__file__).resolve().parent / "results" / "phase5" / "power_planning.json"
EVIDENCE = "DESIGN_STAGE_PLANNING_NOT_EVIDENCE"


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(var)


def null_se(n_pos: int, n_neg: int) -> float:
    """SE of the AUC under H0: AUC = 0.5 (Mann-Whitney)."""
    return math.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))


def analytic_disease_precision(n_pd: int = 24, n_healthy: int = 8) -> dict:
    se0 = null_se(n_pd, n_healthy)
    mde = 0.5 + (Z_ALPHA + Z_POWER) * se0
    rows = []
    for a in (0.6, 0.7, 0.8, 0.9):
        se = hanley_mcneil_se(a, n_pd, n_healthy)
        rows.append({"true_auc": a, "se": se, "ci95_half_width": Z_ALPHA * se, "ci95": [max(0.0, a - Z_ALPHA * se), min(1.0, a + Z_ALPHA * se)]})
    return {"independent_units": {"pd_subjects": n_pd, "healthy_subjects": n_healthy},
            "assumptions": ["one independent score per subject (subject-level aggregation)", "scores treated as fixed: training variability, model selection and "
                            "repeated recordings are ignored, so every number here is OPTIMISTIC", "normal approximation (Hanley-McNeil); poor with 8 healthy subjects"],
            "se_under_null": se0, "minimum_detectable_auc_vs_0_5_at_80pct_power": mde,
            "reading": f"with {n_pd} PD and {n_healthy} healthy subjects an AUC below about {mde:.2f} cannot be reliably distinguished from 0.5, even optimistically",
            "precision_at_assumed_true_auc": rows}


def prevalence_sensitivity(prevalences=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75)) -> dict:
    """HISTORICAL shipped-model operating points (reconstructed from metadata; not reproduced from data)."""
    ops = {"disease_RandomForest_Top10Features": (130 / 147, 21 / 48), "risk_LogisticRegression": (1325 / 2215, 2332 / 3586)}
    out = {}
    for name, (se, sp) in ops.items():
        out[name] = {"sensitivity": se, "specificity": sp, "by_prevalence": [
            {"prevalence": p, "ppv": se * p / (se * p + (1 - sp) * (1 - p)), "npv": sp * (1 - p) / (sp * (1 - p) + (1 - se) * p)} for p in prevalences]}
    return {"evidence_status": "ARITHMETIC_ON_HISTORICAL_UNVERIFIED_FIGURES", "operating_points": out,
            "reading": "PPV collapses as prevalence falls; the historical operating points say nothing about any real screening population"}


def simulate_risk_voice_power(levels=(0.0, 0.25, 0.5, 1.0, 2.0), reps: int = 20, n_boot: int = 200, seed0: int = 9_000) -> dict:
    def synth(seed, voice_signal):
        return P.synthetic_risk(seed, n_patients=42, visits=139, voice_signal=voice_signal)

    rows = []
    for li, lev in enumerate(levels):
        t0 = time.time()
        res = [ST.risk_rep(seed0 + 1000 * li + i, lev, n_boot, synth) for i in range(reps)]
        diffs = [r["diff"] for r in res]
        pos = [r["lo"] is not None and r["lo"] > 0 for r in res]
        neg = [r["hi"] is not None and r["hi"] < 0 for r in res]
        rows.append({"voice_signal_generator_units": lev, "reps": reps, "mean_auc_difference_full_minus_time_baseline": float(np.mean(diffs)),
                     "sd_auc_difference": float(np.std(diffs, ddof=1)), "detection_rate_interval_above_0": float(np.mean(pos)),
                     "interval_below_0_rate": float(np.mean(neg)), "seconds": round(time.time() - t0, 1)})
    ok = [r for r in rows if r["detection_rate_interval_above_0"] >= 0.8]
    return {"design": {"patients": 42, "sessions_per_patient": 139, "outer": 5, "inner": 3, "n_boot": n_boot, "candidates": "LR_C0.1, LR_C1",
                       "note": "a lighter configuration than the registered 10-fold/5-seed protocol; single-seed per repetition"},
            "rows": rows,
            "smallest_tested_effect_with_80pct_detection": None if not ok else {k: ok[0][k] for k in ("voice_signal_generator_units", "mean_auc_difference_full_minus_time_baseline")},
            "caveats": ["the generator's voice effect is arbitrary: the transferable quantity is the AUC difference, not the generator unit",
                        "real labels derive from interpolated UPDRS, real voice is correlated within patient, and the real noise structure is unknown: "
                        "real power can be lower or higher", f"{reps} repetitions per level: detection rates carry roughly +/-{100 * 0.98 / math.sqrt(reps):.0f} points of Monte-Carlo error"]}


def simulate_voice_sample_size(patient_counts=(42, 84, 168), levels=(0.5, 1.0), reps: int = 12, n_boot: int = 150, seed0: int = 31_000) -> dict:
    """How does the chance that the paired interval of FULL minus TIME_BASELINE excludes 0 grow with the number of PATIENTS, for a fixed
    generated voice effect?  Sessions per patient are held at 139.  DESIGN-STAGE PLANNING under an invented generator, not evidence."""
    rows = []
    for li, lev in enumerate(levels):
        for ni, n in enumerate(patient_counts):
            t0 = time.time()

            def synth(seed, voice_signal, _n=n):
                return P.synthetic_risk(seed, n_patients=_n, visits=139, voice_signal=voice_signal)

            res = [ST.risk_rep(seed0 + 10_000 * li + 1000 * ni + i, lev, n_boot, synth) for i in range(reps)]
            diffs = [r["diff"] for r in res]
            pos = [r["lo"] is not None and r["lo"] > 0 for r in res]
            rows.append({"voice_signal_generator_units": lev, "patients": n, "sessions_per_patient": 139, "reps": reps,
                         "mean_auc_difference_full_minus_time_baseline": float(np.mean(diffs)), "sd_auc_difference": float(np.std(diffs, ddof=1)),
                         "detection_rate_interval_above_0": float(np.mean(pos)), "seconds": round(time.time() - t0, 1)})
    return {"design": {"outer": 5, "inner": 3, "n_boot": n_boot, "candidates": "LR_C0.1, LR_C1", "single_reference": "TIME_BASELINE only (the registered rule also needs TIME_BASELINE_DEMOGRAPHIC and every seed)"},
            "rows": rows,
            "caveats": ["power under the registered intersection rule (two references x five seeds) is LOWER than shown",
                        "the generated voice signal is independent across sessions within a patient given the label trend; real voice is autocorrelated",
                        f"{reps} repetitions per cell: detection rates carry roughly +/-{100 * 0.98 / math.sqrt(reps):.0f} points of Monte-Carlo error",
                        "the transferable quantity is the AUC difference; the generator unit is arbitrary",
                        "an effect smaller than the smallest simulated one needs more patients than the largest cell here: extrapolation is not supported"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--skip-simulation", action="store_true")
    ap.add_argument("--sample-size", action="store_true", help="run ONLY the patients-vs-detection simulation and write it to --out")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.sample_size:
        body = {"task": "voice_sample_size_planning", "status": "completed", "evidence_status": EVIDENCE, "clinically_validated": False,
                "protocol_id": P.PROTOCOL["id"], "protocol_sha256": P.PROTOCOL_SHA256, "simulation": simulate_voice_sample_size(reps=a.reps)}
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(body, indent=2) + "\n")
        print(out)
        return 0
    body = {"task": "power_planning", "status": "completed", "evidence_status": EVIDENCE, "clinically_validated": False,
            "protocol_id": P.PROTOCOL["id"], "protocol_sha256": P.PROTOCOL_SHA256,
            "analytic_disease_precision": analytic_disease_precision(), "prevalence_sensitivity": prevalence_sensitivity(),
            "risk_voice_value_simulation": None if a.skip_simulation else simulate_risk_voice_power(reps=a.reps),
            "multiplicity": {"registered_voice_rule": "two reference models (TIME_BASELINE, TIME_BASELINE_DEMOGRAPHIC) x every seed must all exclude 0: "
                                                      "an intersection rule, which is conservative against false positives but costs power",
                             "not_adjusted": "feature-set comparisons beyond the registered rule, subgroup analyses and any additional endpoint are exploratory"}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
