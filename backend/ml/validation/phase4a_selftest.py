"""
Phase 4A harness-integrity study on SYNTHETIC data.  Output is labelled SYNTHETIC_FIXTURE_NOT_EVIDENCE and must never be quoted
as evidence about any model or data set.  Purpose: show that the evaluation machinery itself stays honest under the exact traps it
exists to defeat, with ground truth known by construction.

  A. disease-like, subjects carry strong personal "fingerprints", labels are group-level and carry NO information about features.
       * a naive row-level K-fold must look excellent (the trap works)           -> proves the trap is real
       * the protocol (subject-grouped nested LOSO) must NOT look excellent
       * the subject-label permutation test must reject at about its nominal rate (<= ~5-10%)
  B. disease-like with a planted group-level signal: the protocol must detect it (so the null result above is not just "blind").
  C. risk-like, labels driven by time and baseline only: the registered voice-value interval FULL - TIME_BASELINE must usually
       contain 0 when voice carries no signal ...
  D. ... and usually exclude 0 when voice truly carries signal.

Run:  python -m ml.validation.phase4a selftest
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evalkit import (auc_stat, group_label_permutation_test, leave_one_group_out, nested_group_cv,
                        paired_cluster_bootstrap_diff)
from ml.evalkit.warnings_util import quiet
from ml.validation import candidates as C
from ml.validation import data as D



def lr_cands() -> dict:
    return {"LR_C0.1": C._lr(0.1), "LR_C1": C._lr(1.0)}


def row_level_auc(X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    """The WRONG protocol (kept as the trap): rows of one subject on both sides of the split."""
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))])
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return float(roc_auc_score(y, oof))


def disease_rep(seed: int, signal: float, n_perm: int, synth) -> dict:
    df, _ = synth(seed, signal=signal)
    X = df[D.DISEASE_FEATURES].reset_index(drop=True)
    y, g = df["status"].reset_index(drop=True), df["subject"].reset_index(drop=True)
    nested = nested_group_cv(X, y, g, lr_cands(), outer="logo", inner_k=4, seed=seed)
    obs = auc_stat(pd.DataFrame({"y": y, "p": nested.oof}))

    def p_fn(yp):
        oof = np.zeros(len(yp))
        for tr, te in leave_one_group_out(g, X=X):
            m = lr_cands()["LR_C1"]()
            m.fit(X.iloc[tr], yp[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        return oof

    fixed_obs = auc_stat(pd.DataFrame({"y": y, "p": p_fn(y.to_numpy())}))
    perm = group_label_permutation_test(g, y.to_numpy(), p_fn, fixed_obs, n_perm=n_perm, seed=seed)
    return {"row_level_auc": row_level_auc(X, y, seed), "nested_subject_level_auc": obs, "perm_p": perm["p_value"]}


def risk_rep(seed: int, voice_signal: float, n_boot: int, synth) -> dict:
    df, dcols, _ = synth(seed, voice_signal=voice_signal)
    y, g = df["worsening"].reset_index(drop=True), df["subject#"].astype(str).reset_index(drop=True)
    oofs = {}
    for name, cols in {"FULL": dcols + ["test_time", "baseline_total_UPDRS"], "TIME_BASELINE": ["test_time", "baseline_total_UPDRS"]}.items():
        oofs[name] = nested_group_cv(df[cols].reset_index(drop=True), y, g, lr_cands(), outer=5, inner_k=3, seed=seed,
                                     group_level_target=False).oof
    fr = pd.DataFrame({"subject": g, "y": y, "a": oofs["FULL"], "b": oofs["TIME_BASELINE"]})
    ci = paired_cluster_bootstrap_diff(fr, "subject", "a", "b", n_boot=n_boot, seed=seed)
    return {"diff": ci["point"], "lo": ci.get("lo"), "hi": ci.get("hi")}


@quiet
def run_selftest(reps: int = 24, n_perm: int = 40, n_boot: int = 200, seed0: int = 4_000, scenarios=("A", "B", "B2", "C", "D")) -> dict:
    from ml.validation import phase4a as P
    from ml.validation.phase4a import synthetic_disease, synthetic_risk

    raw: dict = {}
    if "A" in scenarios:
        raw["A"] = [disease_rep(seed0 + i, 0.0, n_perm, synthetic_disease) for i in range(reps)]
    if "B" in scenarios:   # planted effect of 1.5 on 3 of 22 features, subject fingerprints sd 2.0: a HARD scenario (low power expected)
        raw["B"] = [disease_rep(seed0 + 500 + i, 1.5, n_perm, synthetic_disease) for i in range(reps)]
    if "B2" in scenarios:  # a strong effect (4.0), added AFTER B showed low sensitivity; B is still reported as run
        raw["B2"] = [disease_rep(seed0 + 800 + i, 4.0, n_perm, synthetic_disease) for i in range(reps)]
    if "C" in scenarios:
        raw["C"] = [risk_rep(seed0 + 1000 + i, 0.0, n_boot, synthetic_risk) for i in range(reps)]
    if "D" in scenarios:
        raw["D"] = [risk_rep(seed0 + 1500 + i, 3.0, n_boot, synthetic_risk) for i in range(reps)]

    mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
    rate = lambda rows, f: float(np.mean([f(r) for r in rows]))
    rej = lambda r: r["perm_p"] is not None and r["perm_p"] <= 0.05
    out = {"task": "harness_integrity_selftest", "status": "completed", "evidence_status": "SYNTHETIC_FIXTURE_NOT_EVIDENCE",
           "clinically_validated": False, "protocol_id": P.PROTOCOL["id"], "reps_per_scenario": reps, "n_perm": n_perm,
           "n_boot": n_boot, "seed0": seed0, "raw_repetitions": raw}
    if "A" in raw:
        out["A_null_labels_strong_subject_fingerprints"] = {
            "mean_row_level_auc_the_trap": mean(raw["A"], "row_level_auc"),
            "mean_protocol_subject_level_auc": mean(raw["A"], "nested_subject_level_auc"),
            "permutation_rejection_rate_at_0.05": rate(raw["A"], rej)}
    for key, label in (("B", "B_planted_signal_hard"), ("B2", "B2_planted_signal_strong")):
        if key in raw:
            out[label] = {"mean_protocol_subject_level_auc": mean(raw[key], "nested_subject_level_auc"),
                          "permutation_rejection_rate_at_0.05": rate(raw[key], rej)}
    if "C" in raw:
        c = raw["C"]
        out["C_risk_voice_is_noise"] = {
            "mean_auc_difference_full_minus_time_baseline": mean(c, "diff"),
            "ci_excludes_zero_on_the_POSITIVE_side_rate (false 'voice helps' claims)": rate(c, lambda r: r["lo"] is not None and r["lo"] > 0),
            "ci_excludes_zero_on_the_NEGATIVE_side_rate (noise features hurt)": rate(c, lambda r: r["hi"] is not None and r["hi"] < 0),
            "ci_contains_zero_rate": rate(c, lambda r: r["lo"] is not None and r["lo"] <= 0 <= r["hi"])}
    if "D" in raw:
        d = raw["D"]
        out["D_risk_voice_has_signal"] = {"mean_auc_difference_full_minus_time_baseline": mean(d, "diff"),
                                          "ci_excludes_zero_positive_rate (power)": rate(d, lambda r: r["lo"] is not None and r["lo"] > 0)}
    return out
