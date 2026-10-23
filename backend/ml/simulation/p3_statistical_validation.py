"""
Phase 3 independent statistical validation (ALL SYNTHETIC -- none of this is evidence about real patients or real data).

    cd backend && python -m ml.simulation.p3_statistical_validation --out ml/simulation/results/p3_statistical_validation.json

Each section replicates or tests a claim the repository makes, with a seed that differs from the Phase 2 study (20261024)
and with confidence intervals for every estimated rate (Wilson 95% for proportions).

  S1  Student-t prediction interval: false-positive rate vs n under iid Gaussian, AR(1) 0.6 and Student-t(3) noise.
  S2  Self-starting CUSUM/EWMA: family-wise false alarms over 40 readings (re-computes the figures that the Phase 2
      harness only quoted as literals) and a fixed-baseline comparison.
  S3  Mann-Kendall + Theil-Sen trend rule: type-I error with and without the lag-1 variance correction.
  S4  Leakage: row-level vs subject-level cross-validation on data with NO signal; naive best-of-K selection vs nested.
  S5  class_weight='balanced' probabilities: miscalibration, and the analytic prior-shift correction.
  S6  Input-domain (OOD) detector: false-rejection and detection rates on a synthetic correlated Gaussian domain.
  S7  Consensus: does the support index distinguish independent from shared-error evidence?  (It cannot.)
  S8  Risk heuristic: exhaustive enumeration of the weighted-sum scores and bucket map.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import warnings
from pathlib import Path

import numpy as np

SEED = 20261024


def wilson(k: int, n: int) -> list[float]:
    from ml.evalkit.metrics import wilson_interval
    lo, hi = wilson_interval(k, n)
    return [round(lo, 4), round(hi, 4)]


def rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4), "wilson95": wilson(int(k), int(n))}


def noise(kind: str, n: int, rng) -> np.ndarray:
    if kind == "iid_gaussian":
        return rng.standard_normal(n)
    if kind == "iid_t3":
        return rng.standard_t(3, n) / np.sqrt(3.0)
    rho = float(kind.split("_")[1])
    e = rng.standard_normal(n) * np.sqrt(1 - rho * rho)
    z = np.empty(n); z[0] = rng.standard_normal()
    for i in range(1, n):
        z[i] = rho * z[i - 1] + e[i]
    return z


# ------------------------------------------------------------------------------------------------------------- S1
def s1_point_interval(trials: int) -> dict:
    from app.ml.personalized_anomaly import detect_personal_anomaly
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for kind in ("iid_gaussian", "ar1_0.6", "iid_t3"):
        out[kind] = {}
        for n in (3, 4, 5, 8, 15, 30):
            hits = 0
            for _ in range(trials):
                x = 10 + 2 * noise(kind, n + 1, rng)
                hits += detect_personal_anomaly(list(x[:-1]), float(x[-1]))["is_anomaly"]
            out[kind][f"n={n}"] = rate(hits, trials)
    return {"nominal": 0.05, "trials_per_cell": trials, "false_positive_rate": out,
            "reading": "iid Gaussian should sit at 5% for every n (verified if each Wilson interval contains 0.05)"}


# ------------------------------------------------------------------------------------------------------------- S2
def s2_sequential(reps: int) -> dict:
    from app.ml import longitudinal as L
    rng = np.random.default_rng(SEED + 2)
    N = 40
    fam = {"cusum": 0, "ewma": 0, "either": 0}
    for _ in range(reps):
        q = [v for v in L.q_statistics(10 + 2 * rng.standard_normal(N)) if v is not None]
        c = L.cusum(q, 0.5, 5.0)["first_alarm_index"] is not None
        e = L.ewma(q, 0.2, 3.0)["first_alarm_index"] is not None
        fam["cusum"] += c; fam["ewma"] += e; fam["either"] += (c or e)
    self_start = {k: rate(v, reps) for k, v in fam.items()}
    # fixed-baseline comparison: the first b readings estimate mean/SD, the following 35 are monitored with textbook limits
    fixed = {}
    for b in (5, 12, 30):
        hits = {"cusum": 0, "ewma": 0}
        for _ in range(reps):
            x = rng.standard_normal(b + 35)
            z = (x[b:] - x[:b].mean()) / x[:b].std(ddof=1)
            hits["cusum"] += L.cusum(z, 0.5, 5.0)["first_alarm_index"] is not None
            hits["ewma"] += L.ewma(z, 0.2, 3.0)["first_alarm_index"] is not None
        fixed[f"b={b}"] = {k: rate(v, reps) for k, v in hits.items()}
    return {"reps": reps, "series_length": N,
            "self_starting_family_wise_false_alarm_iid_gaussian_no_autocorr_gate": self_start,
            "fixed_baseline_family_wise_false_alarm_35_steps_iid_gaussian": fixed,
            "note": "Phase 2 quoted 6.9%/7.0% (self-starting) and 30-65% (fixed baseline) as literals inside anomaly_study.py; "
                    "this section recomputes them from code."}


# ------------------------------------------------------------------------------------------------------------- S3
def s3_trend(reps: int) -> dict:
    from app.ml import longitudinal as L
    rng = np.random.default_rng(SEED + 3)
    out = {}
    for kind in ("iid_gaussian", "ar1_0.3", "ar1_0.6"):
        out[kind] = {}
        for n in (20, 40):
            t = np.arange(n, dtype=float)
            k_corr = k_raw = 0
            for _ in range(reps):
                y = 10 + 2 * noise(kind, n, rng)
                ts = L.theil_sen(t, y, 0.01)
                resid = y - (ts["intercept"] + ts["slope_per_day"] * t)
                k_corr += L.mann_kendall(y, residuals=resid)["p_value"] < 0.01
                k_raw += L.mann_kendall(y, autocorr_correct=False)["p_value"] < 0.01
            out[kind][f"n={n}"] = {"nominal_alpha": 0.01, "with_autocorr_correction": rate(k_corr, reps),
                                   "without_correction": rate(k_raw, reps)}
    # power for a drift of 3 SD over the series (slope 3*2/(n-1) per reading) -- p-value only (no effect-size rule)
    power = {}
    for n in (20, 40):
        t = np.arange(n, dtype=float); k = 0
        for _ in range(reps):
            y = 10 + 2 * rng.standard_normal(n) + 6.0 * t / (n - 1)
            ts = L.theil_sen(t, y, 0.01)
            k += L.mann_kendall(y, residuals=y - (ts["intercept"] + ts["slope_per_day"] * t))["p_value"] < 0.01
        power[f"n={n}"] = rate(k, reps)
    return {"reps": reps, "type_I_error": out, "power_drift_3sd_total_iid": power}


# ------------------------------------------------------------------------------------------------------------- S4
def s4_leakage(reps: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, KFold
    from ml.evalkit import splits as S
    rng = np.random.default_rng(SEED + 4)
    row_auc, grp_auc = [], []
    for _ in range(reps):
        subj, rows = 30, 12
        g = np.repeat(np.arange(subj), rows)
        label = rng.integers(0, 2, subj)[g]                            # coin flip per subject: NO signal at all
        X = rng.normal(size=(subj, 5))[g] + 0.3 * rng.normal(size=(len(g), 5))   # features identify the SUBJECT, not the label
        for name, splitter, store in (("row", KFold(5, shuffle=True, random_state=0), row_auc),
                                      ("group", GroupKFold(5), grp_auc)):
            oof = np.zeros(len(g))
            for tr, te in splitter.split(X, label, g):
                m = RandomForestClassifier(60, random_state=0, n_jobs=1).fit(X[tr], label[tr])
                oof[te] = m.predict_proba(X[te])[:, 1]
            store.append(roc_auc_score(label, oof))
    # the guard must refuse the row-level split
    refused = False
    try:
        g = np.repeat(np.arange(10), 4)
        list(S.guarded(KFold(3, shuffle=True, random_state=0).split(g, groups=g), groups=g))
    except S.LeakageError:
        refused = True

    # selection optimism on pure noise. Candidates = single-feature or all-feature logistic models (genuinely different
    # hypotheses, so the best of K is optimistic by chance). Naive and nested use the SAME outer grouped 6-fold split.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from ml.evalkit.nested import nested_group_cv
    import pandas as pd
    cols = list("abcdef")
    def mk(sel):
        return lambda: Pipeline([("sel", FunctionTransformer(lambda X, sel=sel: X[sel])), ("s", StandardScaler()),
                                 ("c", LogisticRegression(max_iter=500))])
    cand = {f"only_{c}": mk([c]) for c in cols}
    cand["all"] = mk(cols)
    naive_best, nested, logo_pooled = [], [], []
    for r in range(max(6, reps // 3)):
        subj = 24
        g = np.repeat(np.arange(subj), 3)
        y = rng.integers(0, 2, subj)[g]
        y[:3] = 0; y[-3:] = 1
        X = pd.DataFrame(rng.normal(size=(subj, 6))[g] + 0.5 * rng.normal(size=(len(g), 6)), columns=cols)
        pairs = list(S.group_kfold(g, y, n_splits=6, seed=r, X=X, check_duplicates=False))
        aucs = []
        for name, f in cand.items():
            oof = np.zeros(len(y))
            for tr, te in pairs:
                oof[te] = f().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y, oof))
        naive_best.append(max(aucs))
        nested.append(roc_auc_score(y, nested_group_cv(X, y, g, cand, outer=6, inner_k=3, seed=r).oof))
        oof = np.zeros(len(y))                                   # leave-one-subject-out, single model, pooled
        for tr, te in S.leave_one_group_out(g, X=X, check_duplicates=False):
            oof[te] = cand["all"]().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        logo_pooled.append(roc_auc_score(y, oof))
    ci = lambda a: [round(float(np.mean(a)), 4), round(float(np.mean(a) - 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))), 4),
                    round(float(np.mean(a) + 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))), 4)]
    return {"reps": reps, "truth": "labels are independent of features (true AUC = 0.5)",
            "row_level_cv_auc_mean_lo_hi": ci(row_auc), "subject_level_cv_auc_mean_lo_hi": ci(grp_auc),
            "leakage_guard_refuses_row_split": refused,
            "selection_reps": len(naive_best), "naive_best_of_7_auc_mean_lo_hi": ci(naive_best),
            "nested_auc_mean_lo_hi": ci(nested),
            "leave_one_subject_out_pooled_auc_on_null_mean_lo_hi": ci(logo_pooled),
            "logo_reading": "pooled leave-one-subject-out AUC on no-signal data falls BELOW 0.5 (training prior shifts against the held-out "
                            "subject), so a LOSO result cannot be read against a 0.5 'chance' line without this correction"}


# ------------------------------------------------------------------------------------------------------------- S5
def s5_calibration() -> dict:
    from sklearn.linear_model import LogisticRegression
    from ml.evalkit.metrics import expected_calibration_error
    rng = np.random.default_rng(SEED + 5)
    out = {}
    for prev in (0.05, 0.10, 0.25, 0.50, 0.75):
        n = 40000
        y = (rng.random(n) < prev).astype(int); X = rng.normal(size=(n, 1)) + 1.0 * y[:, None]
        Xt = rng.normal(size=(n, 1)); yt = (rng.random(n) < prev).astype(int); Xt = Xt + 1.0 * yt[:, None]
        row = {}
        for name, kw in (("plain", {}), ("balanced", {"class_weight": "balanced"})):
            p = LogisticRegression(**kw).fit(X, y).predict_proba(Xt)[:, 1]
            row[name] = {"mean_predicted": round(float(p.mean()), 4), "ece": round(expected_calibration_error(yt, p), 4)}
        # analytic correction: odds_true = odds_balanced * prev/(1-prev)
        p = LogisticRegression(class_weight="balanced").fit(X, y).predict_proba(Xt)[:, 1]
        o = p / (1 - p) * prev / (1 - prev); pc = o / (1 + o)
        row["balanced_after_prior_shift_correction"] = {"mean_predicted": round(float(pc.mean()), 4),
                                                        "ece": round(expected_calibration_error(yt, pc), 4)}
        out[f"prevalence={prev}"] = row
    return {"model": "logistic regression, one feature, true shift 1 SD (well specified)", "results": out,
            "reading": "balanced weighting makes scores ~ posterior at a 50% prior; the correction needs the DEPLOYMENT prevalence, "
                       "which this project does not know -- hence scores are labelled uncalibrated (D10)."}


# ------------------------------------------------------------------------------------------------------------- S6
def s6_ood(reps: int) -> dict:
    import pandas as pd
    from app.ml.ood import evaluate_domain_detector, fit_domain
    rng = np.random.default_rng(SEED + 6)
    p = 8
    A = rng.normal(size=(p, p)); cov = A @ A.T / p + np.eye(p) * 0.5
    chol = np.linalg.cholesky(cov)
    cols = [f"f{i}" for i in range(p)]
    def draw(n): return pd.DataFrame(rng.normal(size=(n, p)) @ chol.T + 5.0, columns=cols)
    groups = np.repeat(np.arange(25), 20)
    res = {"p_features": p, "n_train_rows": 500, "n_train_groups": 25, "reps": reps, "rows": []}
    fr, miss = [], {"shift_3sd_one_feature": [], "unit_error_x1000_one_feature": [], "break_correlation_shuffle_pair": [], "all_features_shift_1sd": []}
    for r in range(reps):
        dom = fit_domain("generic", draw(500), groups=groups, range_margin=0.10)
        test = draw(400)
        rows = [dict(zip(cols, map(float, v))) for v in test.to_numpy()]
        sd = test.std().to_numpy()
        def f_shift(d): d = dict(d); d["f0"] += 3 * sd[0]; return d
        def f_unit(d): d = dict(d); d["f1"] *= 1000; return d
        def f_corr(d):                                   # keep both marginals plausible but break their joint structure
            d = dict(d); d["f2"], d["f3"] = 5.0 + (d["f2"] - 5.0) * 0.0 + sd[2], 5.0 - sd[3]; return d
        def f_all(d): return {k: v + sd[i] for i, (k, v) in enumerate(d.items())}
        ev = evaluate_domain_detector(dom, rows, {"shift_3sd_one_feature": f_shift, "unit_error_x1000_one_feature": f_unit,
                                                  "break_correlation_shuffle_pair": f_corr, "all_features_shift_1sd": f_all})
        fr.append(ev["false_rejection_rate"])
        for k in miss: miss[k].append(ev["perturbations"][k]["missed_ood_rate"])
    res["false_rejection_rate_mean"] = round(float(np.mean(fr)), 4)
    res["false_rejection_rate_range"] = [round(float(min(fr)), 4), round(float(max(fr)), 4)]
    res["missed_ood_rate_mean"] = {k: round(float(np.mean(v)), 4) for k, v in miss.items()}
    res["design_target_false_rejection"] = "<= 1% from the 0.99 quantile threshold, plus range-margin rejections"
    return res


# ------------------------------------------------------------------------------------------------------------- S7
def s7_consensus(trials: int) -> dict:
    from app.core.consensus import compute_evidence_consensus
    from app.core.evidence import Direction, EvidenceType, HypothesisEvidence
    from app.schemas.agent import AgentResult, AgentStatus
    rng = np.random.default_rng(SEED + 7)
    W = {"a": 0.65, "b": 0.65}
    def res(agent, ev): return AgentResult(agent_name=agent, task_id="t", status=AgentStatus.COMPLETED, hypothesis_evidence=ev)
    def ev(agent, h, s, grp=""): return HypothesisEvidence(source_agent=agent, hypothesis=h, evidence_type=EvidenceType.DIAGNOSTIC,
                                                            direction=Direction.SUPPORTS, strength=float(np.clip(s, 0, 1)), independence_group=grp)
    def trial(mode):
        # truth = H1 (support 0.55) vs distractor H2 (support 0.45) ; observation noise SD 0.15
        e1 = rng.normal(0, 0.15, 2); e2 = rng.normal(0, 0.15, 2)
        if mode == "shared":               # both agents read the SAME corrupted input -> identical noise, but each declares its own group
            e1[1], e2[1] = e1[0], e2[0]
        results = [res("a", [ev("a", "H1", 0.55 + e1[0]), ev("a", "H2", 0.45 + e2[0])])]
        if mode != "single":
            results.append(res("b", [ev("b", "H1", 0.55 + e1[1]), ev("b", "H2", 0.45 + e2[1])]))
        out = compute_evidence_consensus(results, W)
        srcs = max((h.independent_sources for h in out.hypotheses), default=0)
        return out.top_hypothesis == "H1", srcs
    summary = {}
    for mode in ("single", "independent", "shared"):
        right = srcs = 0
        for _ in range(trials):
            ok, s = trial(mode); right += ok; srcs += s
        summary[mode] = {"top1_correct": rate(right, trials), "reported_independent_sources": round(srcs / trials, 2)}
    return {"trials": trials, "setup": "true H1 support .55 vs H2 .45, N(0,.15) observation noise, 2 voting agents",
            "results": summary,
            "reading": "'shared' reports 2 independent sources like 'independent' but its accuracy equals the single-source accuracy: "
                       "the index cannot see correlated evidence unless the emitter declares it (known limitation)."}


# ------------------------------------------------------------------------------------------------------------- S8
def s8_risk_enumeration() -> dict:
    from app.agents.risk import _bucket
    def bucket_of(x): return _bucket(round(x, 2))
    fall_terms = {"gait_slowing": 0.4, "tremor_increase": 0.25, "longitudinal_deteriorating": 0.25, "missed_checkins_ge3": 0.1}
    fall = {}
    for combo in itertools.product([0, 1], repeat=4):
        s = min(sum(w for on, w in zip(combo, fall_terms.values()) if on), 1.0)
        fall["+".join(k for on, k in zip(combo, fall_terms) if on) or "none"] = {"score": round(s, 2), "bucket": bucket_of(s)}
    seiz = {}
    for low_adh, recent, urg in itertools.product([0, 1], [0, 1, 2, 3, 5], [0, 1]):
        s = min(0.35 * low_adh + min(0.1 * recent, 0.4) + 0.3 * urg, 1.0)
        seiz[f"low_adherence={low_adh},recent_seizures={recent},urgency_high={urg}"] = {"score": round(s, 2), "bucket": bucket_of(s)}
    det = {}
    for lg, dx in itertools.product([0, 1], [0, 1]):
        s = 0.5 * lg + 0.25 * dx
        det[f"longitudinal_deteriorating={lg},diagnostic_conf_gt_0.5={dx}"] = {"score": s, "bucket": bucket_of(s)}
    return {"fall_risk": fall, "seizure_risk": seiz, "deterioration_risk": det,
            "observations": [
                "Every weight is an unvalidated constant; the bucket is a monotone step function of a weighted sum of binary flags.",
                "The same longitudinal 'deteriorating' flag raises BOTH fall_risk (+.25) and deterioration_risk (+.50): the sub-scores are not independent.",
                "overall = max of the three sub-scores, so one saturated factor dominates regardless of the others.",
                "deterioration_risk adds the diagnostic agent's retrieval confidence (a similarity score) as if it measured clinical deterioration.",
                "A single missing input (e.g. no adherence data) lowers the score exactly like a good value; coverage is reported separately."]}


def main():
    import warnings as _w
    _w.simplefilter("ignore")   # scoped to this script's own run

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all repetition counts (use 0.1 for a smoke run)")
    a = ap.parse_args()
    k = lambda n: max(10, int(n * a.scale))
    out = {"seed_base": SEED, "python": platform.python_version(), "numpy": np.__version__, "synthetic": True,
           "S1_point_prediction_interval": s1_point_interval(k(20000)),
           "S2_sequential_charts": s2_sequential(k(4000)),
           "S3_trend_rule": s3_trend(k(1500)),
           "S4_leakage_and_selection": s4_leakage(k(40)),
           "S5_class_weight_calibration": s5_calibration(),
           "S6_input_domain_detector": s6_ood(k(20)),
           "S7_consensus_shared_error": s7_consensus(k(3000)),
           "S8_risk_heuristic_enumeration": s8_risk_enumeration()}
    text = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
