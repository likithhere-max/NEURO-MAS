# NEURO-MAS — Mathematical Specification (Deliverable 3)

> **Point-in-time record (Phase 1 audit, 2026-10-23, commit `1be5e52`).** Parts of this document have been superseded by Phase 2 — see `docs/phase2/MATHEMATICAL_VALIDATION.md` and `docs/phase2/OUTSTANDING_ISSUES.md`. Where the two disagree, **the Phase 2 document is current**. Inline notes marked **[Phase 2]** flag changed items.

Every formula below is implemented in the named function and covered by the named tests.
"Verified" means *the code computes this formula* (hand calculation reproduced by a test). It does
**not** mean the formula is clinically valid; that is stated per section.

---

## 1. Hypothesis consensus — `app/core/consensus.py::compute_consensus`

> **[Phase 2]** The orchestrator now uses `compute_evidence_consensus` (typed, independent evidence; uncalibrated support index; no percentage). The text below documents the *legacy* function that remains for compatibility. See `docs/phase2/CONSENSUS_DESIGN.md`.

For hypothesis H with voting-agent set V(H):

    C(H) = Σ_{i∈V'(H)} w_i · q_i · e_i · s_i(H)  /  Σ_{i∈V'(H)} w_i · q_i
    D    = 1 − max_H C(H)                 (disagreement)
    margin = C(top1) − C(top2)

| Symbol | Meaning | Source |
|---|---|---|
| w_i | agent reliability weight (informal constants: diagnostic 0.65, medication 0.85, …) | `BaseNeuroAgent.reliability_weight` — **not validated** |
| q_i | input data quality ∈[0,1] (state completeness; falls back to the agent's confidence if 0) | `data_intake` |
| s_i(H) | agent i's support for H ∈[0,1], **hypothesis-specific** | caller (`hypothesis_support[H][agent]`) |
| e_i | evidence strength; **1.0 when s_i is hypothesis-specific** | see below |
| V'(H) | voters that completed **and** supplied s_i(H); others *abstain* | `abstained_agents` |

Action is permitted iff at least one voter exists **and** D ≤ `consensus_disagreement_threshold` (0.4).
Inputs are clipped to [0,1]; NaN/∞ → 0.

**Defects corrected (reproduced on the original code):**
1. s_i was the agent's single global confidence, identical for every hypothesis → all hypotheses tied (0.245 = 0.245), so "consensus" carried no information.
2. e_i defaulted to the same confidence → confidence enters squared (double-counting). With hypothesis-specific support the evidence is already inside s_i.
3. A voter was counted whenever a result object existed, even a FAILED one (using whatever confidence the failure carried). Failed/skipped agents now abstain.

**Worked example.** One voter: diagnostic agent, w=0.65, q=0.8, s=0.6, e=1:
C = (0.65·0.8·0.6)/(0.65·0.8) = **0.6**, D = **0.4** ≤ 0.4 → permitted (boundary case, tested).
Two voters (w,q,s) = (0.65, 0.8, 0.6) and (0.85, 0.8, 0.2): num = 0.312+0.136 = 0.448; den = 0.52+0.68 = 1.2 → C = **0.373**, D = 0.627 > 0.4 → not permitted.

**Currently only `neuro_diagnostic_agent` votes** (the others do not produce per-disease support). So today the "consensus" is effectively the diagnostic agent's retrieval score with its quality weight cancelling out. It is an aggregation *framework*, not a multi-agent agreement measure; the UI text says so.
Validity: NOT clinically validated. Weights are unfitted. Tests: `test_consensus.py`.

---

## 2. Personalised anomaly detection — `app/ml/personalized_anomaly.py`

Given a patient's own history x₁…x_n (finite values only; n ≥ 3 = `MIN_READINGS_FOR_STATS`) and a new reading y:

    z = (y − x̄) / s            s = sample SD (n−1)
    flag iff |z| ≥ t_{0.975, n−1} · √(1 + 1/n)

The right-hand side is the half-width (in units of s) of the 95 % **prediction interval** for one new
observation from a Gaussian i.i.d. sample: (y−x̄)/(s√(1+1/n)) ~ t_{n−1}.

| n | critical \|z\| (this spec) | original fixed \|z\|≥2 rule: exact Gaussian false-positive rate |
|---|---|---|
| 3 | 4.968 | 22.5 % |
| 5 | 3.041 | 14.2 % |
| 10 | 2.373 | 8.9 % |
| 30 | 2.079 | 5.9 % |

(Right-hand column computed analytically: P(\|t_{n−1}\| ≥ 2/√(1+1/n)).)

Monte-Carlo (this session, Gaussian null, after the fix): false-positive rate **0.0518 (n=3), 0.051 (n=5), 0.0498 (n=10), 0.0496 (n=30)** — matches the nominal 5 %. Power for a 4-SD shift at n=10 ≈ 0.92.

**Worked example.** History [10, 12, 11, 13, 9]: x̄=11, s=1.581, t_{0.975,4}=2.776, threshold = 2.776·√1.2 = **3.041**.
Reading 20 → z=5.69 → anomaly. Reading 15 → z=2.53 → *not* flagged (the old fixed-2 rule would have flagged it).

**Edge cases (tested):** non-finite history values dropped; zero variance → tolerance of 5 % relative / 1e-6 absolute instead of dividing by 0; `invalid_input` method for a non-finite reading.
**Validity:** exact only under Gaussian i.i.d. data. Real Parkinson's tremor/gait series are autocorrelated, trending, and medication-state dependent; false-positive rates on real data are **unmeasured**.

---

## 3. Change classification — `app/core/metric_semantics.py::classify_change`

    Δ_abs = current − baseline
    Δ_pct = Δ_abs / |baseline|                (undefined if baseline == 0)
    cmp   = round(Δ_pct, 9)                   (removes IEEE-754 noise: 0.19999999999999996 → 0.2)

With polarity p(metric) ∈ {higher_worse, lower_worse, unknown}:

| Condition | Classification |
|---|---|
| baseline or current non-finite | `invalid` |
| baseline = 0 | `baseline_zero_undefined` |
| |cmp| ≤ 0.05 (STABLE_BAND) | `stable` |
| p unknown | `changed_direction_unknown` |
| movement in the "worse" direction | `deteriorating` (anomaly if cmp ≥ 0.20 `ADVERSE_THRESHOLD`) |
| movement in the "better" direction | `improving` |

Polarity comes from a regex registry (e.g. tremor, falls, UPDRS ⇒ higher_worse; gait speed, MoCA, adherence ⇒ lower_worse). Metrics not in the registry are **never guessed**; they are listed in `unclassified_metrics`.
The 5 % / 20 % thresholds are engineering defaults, **not** clinically derived minimal clinically important differences. Tests: `test_metric_semantics.py`.

---

## 4. Risk score — `app/agents/risk.py`

Weighted-factor heuristic (weights in code). **Not** a validated risk model. What changed: the reported
`confidence` is now *input coverage* = (available inputs)/5 and is labelled so; a low score with few inputs
raises `risk_scores_low_may_reflect_missing_inputs_not_low_risk`.

## 5. Other "confidence" values

| Agent | Meaning of `confidence` after the fix |
|---|---|
| data_intake | fraction of supplied numeric blocks that are valid |
| longitudinal | fraction of metrics that could be classified |
| medication | fraction of supplied medications covered by the 5-rule table |
| knowledge | KB coverage for the retrieved conditions |
| risk | input coverage (see above) |
| diagnostic | top retrieval score (capped), per-hypothesis as `s_i(H)` |
| emergency | 1.0 if a rule fired else 0.0 — a rule indicator, not a probability |

None are calibrated probabilities. Each agent emits a warning saying so.

---

## 6. ML models (demo endpoints) — see `ML_VALIDATION_REPORT.md`
Metrics definitions used by the validation script: leave-one-subject-out (LOSO), subject-level
aggregation (mean of window probabilities per subject), 95 % percentile bootstrap over **subjects**,
Brier score, expected calibration error (10 bins), prevalence-adjusted PPV = Se·π/(Se·π+(1−Sp)(1−π)).
No real-data number is reported anywhere in this repo because the datasets could not be fetched.
