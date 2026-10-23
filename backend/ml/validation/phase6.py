"""
Phase 6 extension runner (protocol `p6-1-ext`, parent `p4a-1`).

Runs the candidate families in `candidates_p6` under the SAME nested subject-grouped design, seeds and fold assignments as the
registered p4a-1 run (folds depend only on subject identity and the seed, never on the features), so the extension's held-out
predictions can be compared with p4a-1's on identical participants.  It does not modify p4a-1, its candidates, its feature sets,
its threshold or its decision rule.

Evidence status: whatever the data authentication says, plus `+EXPLORATORY_EXTENSION_P6_1`.  Families were prespecified (see
`candidates_p6.HYPOTHESES` and `protocol_registry/p6-1.json`) but the extension is NOT the registered confirmatory analysis: a
better result here is a hypothesis for an independent cohort, not a finding.

It is reached through the workflow (so every authentication and refusal gate applies):
    python -m ml.validation.workflow run --candidate-set p6-1-ext --disease-csv ... --risk-csv ... --out ...
"""
from __future__ import annotations

import hashlib
import json
import time

import pandas as pd

from ml.evalkit import nested_group_cv
from ml.validation import candidates_p6 as C6
from ml.validation import phase4a as P

EXT_ID = "p6-1-ext"
PARENT_PROTOCOL_ID = P.PROTOCOL["id"]

# --------------------------------------------------------------------------------------------- frozen Phase 6 prespecification
# ADDITIVE: p4a-1 stays the registered primary analysis, unchanged and separately hashed.  This record fixes, before any real result
# exists, what Phase 6 will compute in addition.  Edit never; a change is a new file (p6-2.json) with the reason.
PROTOCOL_P6 = {
    "id": "p6-1",
    "parent_protocol": {"id": "p4a-1", "sha256": "0b2f10a7ec885419d9a2e72ad80b9d4c60b76987ea71e654edcb157b86723e42",
                        "relationship": "ADDITIVE. p4a-1 is not replaced, re-parameterised or reinterpreted by this record."},
    "analysis_classes": {
        "p4a-1 registered results": "confirmatory-internal (as registered)",
        "A. comprehensive metrics on p4a-1 held-out predictions": "secondary and descriptive; no model is refitted",
        "B. extension candidate families (p6-1-ext)": "EXPLORATORY; hypotheses are written down, results are hypotheses for an independent cohort",
        "C. cross-fitted recalibration": "EXPLORATORY; not a deployable calibration",
        "D. decision curves": "diagnostic only until a clinical action and threshold range are defined by clinicians",
        "E. subgroups": "descriptive; reported only above fixed minimum sizes",
    },
    "threshold": 0.5,
    "threshold_note": "same fixed a-priori value as p4a-1; no operating point is chosen from evaluation data; clinically justified thresholds REQUIRE_CLINICAL_INPUT",
    "metrics": ["roc_auc", "pr_auc", "brier", "log_loss", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "ppv", "npv",
                "calibration_slope", "calibration_intercept"],
    "uncertainty": "cluster (subject) bootstrap, percentile, 95%; paired resamples for model comparisons",
    "pairing_rule": "extension and p4a-1 predictions may be compared only when every (seed, row) has the same outer fold in both runs",
    "extension_feature_sets_risk": ["FULL", "TIME_BASELINE", "TIME_ONLY", "VOICE_ONLY", "TIME_BASELINE_DEMOGRAPHIC"],
    "extension_families": {"risk": ["LR_unweighted_C*", "LR_enet_C*", "LR_spline_C*", "HGB_regularised", "HGB_monotone_time"],
                           "disease": ["LR_unweighted_C*", "LR_enet_C*"]},
    "voice_value_study": {"A_time_only": "TIME_ONLY", "B_time_baseline": "TIME_BASELINE", "C_time_baseline_voice": "FULL", "D_voice_only": "VOICE_ONLY",
                          "decision_rule": "unchanged from p4a-1 (interval of FULL minus each reference excludes 0 for every reference and every seed)",
                          "inconclusive_rule": "if the rule is not met the result is reported INCONCLUSIVE with the additional data estimated by ml.validation.power_planning, not as 'voice has no value'"},
    "decision_curve": {"threshold_grid": [round(0.05 * i, 2) for i in range(1, 20)],
                       "interpretation": "REQUIRES_CLINICAL_INPUT: no clinical action or threshold range is defined"},
    "subgroups": {"risk": ["sex", "age_tertile (cut points from subject ages only)", "baseline_severity_tertile (cut points from first-visit scores only)"],
                  "disease": [], "min_clusters": 8, "min_rows_per_class": 20},
    "recalibration": {"method": "logistic on logit(score), grouped 5-fold across subjects, fitted on other subjects' held-out scores", "label": "EXPLORATORY_NOT_A_DEPLOYABLE_CALIBRATION"},
    "acceptance_criteria": "REQUIRES_CLINICAL_INPUT: no performance target is set by this record",
    "not_estimable_from_public_data": ["missing-voice behaviour (the public files have no missing voice records)", "performance on a new site, device or population"],
}
PROTOCOL_P6_SHA256 = hashlib.sha256(json.dumps(PROTOCOL_P6, sort_keys=True).encode()).hexdigest()
EXPLORATORY_LABEL = "+EXPLORATORY_EXTENSION_P6_1"
RISK_SETS = ("FULL", "TIME_BASELINE", "TIME_ONLY", "VOICE_ONLY", "TIME_BASELINE_DEMOGRAPHIC")   # the p4a-1 voice-value sets


def risk_feature_sets(delta_cols: list[str]) -> dict[str, list[str]]:
    demo = ["age", "sex"]
    full = {"FULL": delta_cols + ["test_time", "baseline_total_UPDRS"], "VOICE_ONLY": delta_cols,
            "TIME_BASELINE": ["test_time", "baseline_total_UPDRS"], "TIME_ONLY": ["test_time"],
            "TIME_BASELINE_DEMOGRAPHIC": ["test_time", "baseline_total_UPDRS"] + demo}
    assert set(full) == set(RISK_SETS) and set(RISK_SETS) <= set(P.PROTOCOL["risk"]["feature_sets"]), "extension sets drifted from p4a-1"
    return full


def run_risk_extension(df: pd.DataFrame, delta_cols: list[str], info: dict, *, seeds: list[int], fast: bool, evidence: str, recorder=None) -> dict:
    t0 = time.time()
    y, g = df["worsening"].reset_index(drop=True), df["subject#"].astype(str).reset_index(drop=True)
    sets = risk_feature_sets(delta_cols)
    missing = sorted({c for cols in sets.values() for c in cols if c not in df.columns})
    if missing:
        raise P.D.DatasetVerificationError(f"risk: columns required by the extension are absent: {missing}")
    out: dict = {}
    for seed in seeds:
        for name, cols in sets.items():
            cands = C6.risk_extension_candidates(cols, fast)
            res = nested_group_cv(df[cols].reset_index(drop=True), y, g, cands, outer=P.PROTOCOL["risk"]["outer"], inner_k=P.PROTOCOL["inner_k"],
                                  seed=seed, group_level_target=False, check_duplicates=name in ("FULL", "VOICE_ONLY"))
            if recorder:
                recorder(("risk", f"{name}|ext"), seed, y, g, res)
            out.setdefault(name, {})[seed] = {"candidates": sorted(cands), "chosen_counts": res.chosen_counts(),
                                              "n_fallback_selections": sum(f.selection_status != "selected" for f in res.folds)}
    return P.envelope("risk_worsening_extension", "completed", evidence + EXPLORATORY_LABEL, dataset=info, seeds=seeds, extension_id=EXT_ID,
                      protocol_id=PROTOCOL_P6["id"], protocol_sha256=PROTOCOL_P6_SHA256, protocol=PROTOCOL_P6,
                      parent_protocol_id=PARENT_PROTOCOL_ID, parent_protocol_sha256=P.PROTOCOL_SHA256,
                      design={"outer": P.PROTOCOL["risk"]["outer"], "inner": f"StratifiedGroupKFold({P.PROTOCOL['inner_k']}) on outer-training rows only",
                              "threshold": P.PROTOCOL["threshold"], "unit_of_resampling": "patient", "feature_sets": {k: v for k, v in sets.items()}},
                      hypotheses=C6.HYPOTHESES, results_by_set=out,
                      note="held-out predictions are saved by the recorder; metrics are produced by ml.validation.phase6_report from those predictions",
                      runtime_seconds=round(time.time() - t0, 1))


def run_disease_extension(df: pd.DataFrame, info: dict, *, seeds: list[int], fast: bool, evidence: str, recorder=None) -> dict:
    t0 = time.time()
    X = df[P.D.DISEASE_FEATURES].reset_index(drop=True)
    y, g = df["status"].reset_index(drop=True), df["subject"].reset_index(drop=True)
    cands = C6.disease_extension_candidates(fast)
    out: dict = {}
    for seed in seeds:
        res = nested_group_cv(X, y, g, cands, outer="logo", inner_k=P.PROTOCOL["inner_k"], seed=seed)
        if recorder:
            recorder(("disease", "ext"), seed, y, g, res)
        out[seed] = {"candidates": sorted(cands), "chosen_counts": res.chosen_counts(),
                     "n_fallback_selections": sum(f.selection_status != "selected" for f in res.folds)}
    return P.envelope("disease_voice_classifier_extension", "completed", evidence + EXPLORATORY_LABEL, dataset=info, seeds=seeds, extension_id=EXT_ID,
                      protocol_id=PROTOCOL_P6["id"], protocol_sha256=PROTOCOL_P6_SHA256, protocol=PROTOCOL_P6,
                      parent_protocol_id=PARENT_PROTOCOL_ID, parent_protocol_sha256=P.PROTOCOL_SHA256,
                      design={"outer": "leave-one-subject-out", "inner": f"StratifiedGroupKFold({P.PROTOCOL['inner_k']}) on outer-training rows only",
                              "threshold": P.PROTOCOL["threshold"], "unit_of_resampling": "subject"},
                      hypotheses=C6.HYPOTHESES, results_by_seed=out,
                      note="held-out predictions are saved by the recorder; metrics are produced by ml.validation.phase6_report from those predictions",
                      runtime_seconds=round(time.time() - t0, 1))
