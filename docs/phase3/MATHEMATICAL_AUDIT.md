# NEURO-MAS Phase 3: Mathematical Audit

Scope: an independent check of every mathematical or statistical component that exists **in the code** on branch
`phase2/advanced-ml-hardening` (HEAD `d4c6ac8` plus the uncommitted Phase 3 changes). Nothing here was assumed from
documentation; each row names the file that implements it.

Status vocabulary (used throughout):

| Status | Meaning |
|---|---|
| **verified** | The implemented formula matches an independently derived equation, and an independent test (hand-worked value, closed form, or a different library) agrees. Says nothing about clinical usefulness. |
| **incorrect** | A demonstrable defect was found. Fixed items are listed in `DEFECTS_AND_FIXES.md`. |
| **incomplete** | Correct as far as it goes, but a known gap or approximation limits what it supports. |
| **unverified** | No independent evidence is possible with what is in the repository (usually needs real data or a clinician). |

Nothing in this audit is evidence of clinical validity, safety or fitness for use. "Verified" means *the arithmetic does what
its specification says*.

---

## 1. Component map

| # | Component | Source | Status |
|---|---|---|---|
| A | Point prediction interval (within-patient outlier) | `app/ml/personalized_anomaly.py` | verified (D07 fixed) |
| B | Quesenberry self-starting Q-statistics | `app/ml/longitudinal.py::q_statistics` | verified |
| C | Tabular CUSUM, EWMA | `app/ml/longitudinal.py::cusum, ewma` | verified; operating characteristics measured |
| D | Mann-Kendall (tie-corrected, lag-1 inflation) | `app/ml/longitudinal.py::mann_kendall` | verified formula; **incomplete** autocorrelation correction |
| E | Theil-Sen slope, MAD scale | `app/ml/longitudinal.py` | verified |
| F | Series cleaning / `analyze_series` orchestration | `app/ml/longitudinal.py` | verified by fuzz + properties |
| G | Baseline-vs-current change classification | `app/core/metric_semantics.py` | verified after D02, D03 |
| H | Metric polarity registry | `app/core/metric_semantics.py` | verified after D03 |
| I | Risk heuristic (fall / seizure / deterioration) | `app/agents/risk.py` | arithmetic verified; weights **unverified** |
| J | Evidence consensus (support index) | `app/core/consensus.py`, `app/core/evidence.py` | verified arithmetic; **incomplete** (independence is declared) |
| K | Legacy `compute_consensus` | `app/core/consensus.py` | unused in app code; not audited further |
| L | Retrieval scoring (lexical + semantic) | `app/services/knowledge_base.py`, `app/rag/knowledge_retriever.py` | verified after D06, D09 |
| M | Drug-interaction severity combination | `app/services/ddi.py`, `app/services/clinical_rules.py` | verified after D04, D05 |
| N | Safety gate | `app/core/safety.py`, orchestrator | verified after D01, D12 |
| O | Signal-analysis threshold checks | `app/agents/signal_analysis.py` | arithmetic verified; thresholds **unverified** |
| P | Adherence, completeness, data quality | `app/services/repository.py`, `app/schemas/patient_state.py` | D08 fixed; others **incomplete** |
| Q | Evaluation metrics (Wilson, ECE, PPV/NPV, F1, AUC, Brier, bootstrap) | `ml/evalkit/metrics.py` | verified |
| R | Input-domain / OOD detector | `app/ml/ood.py` | formulas verified; detection power **measured, weak** |
| S | Trained Parkinson's models, preprocessing, thresholds | `ml/training/*`, `ml/models/*`, `ml/validation/*` | **unverified** (real-data validation blocked) |

---

## 2. Formula-by-formula

### A. Point prediction interval — `personalized_anomaly.py`

For a patient's own history `x_1..x_n` (n ≥ 3, non-finite values dropped and counted) and a new reading `x`:

```
x̄ = (1/n) Σ x_i          s = sqrt( Σ (x_i − x̄)² / (n − 1) )          z = (x − x̄) / s
flag  ⇔  |z| ≥ t_{1−α/2, n−1} · sqrt(1 + 1/n)
```

*Basis.* For iid Gaussian data `(x − x̄) / (s·sqrt(1 + 1/n)) ~ t_{n−1}` exactly, so this is the textbook (1−α) prediction interval
for one new observation.

*Units.* `z` is dimensionless; invariant to affine rescaling of the metric (tested as a property, offsets up to 1e8).

*Implementation details checked.* `t_{.975,df}` for df ≤ 30 comes from a 3-decimal table (max error 4.7e-4 against scipy) and for
df > 30 from a Cornish-Fisher expansion (error < 1e-4); other α use scipy's exact quantile. The decision rule is `≥` (boundary
flagged). Reflection symmetry and history-order independence hold.

*Hand example (test `test_prediction_interval_hand_example`).* History 10,12,11,13,9: x̄ = 11, Σ(x−x̄)² = 10, s = √2.5 = 1.58114,
n = 5 → threshold 2.776·√1.2 = 3.0410. Latest 16 → z = 3.1623 flagged; latest 14 → z = 1.8974 not flagged.

*Measured.* With 2,000,000 iid Gaussian trials per n the false-positive rate is 0.0500, 0.0500, 0.0500, 0.0498, 0.0499, 0.0499,
0.0496 for n = 3, 5, 15, 30, 31, 40, 60 (`ml/simulation/p3_pi_large_check.py`; the |z| versus 0.05 deviations are ≤ 2.4 SE, with the
largest at n = 60). This confirms the threshold is attained for a **single** review of Gaussian data.

*Failure mode found and fixed (D07).* A float-noise SD (2.8e-17 on a constant series) was treated as real, giving z = 3.6e9. The
zero-variance branch now triggers when `s ≤ 1e-12·max(1,|x̄|)`.

*Unsupported assumptions.* iid, Gaussian, no state changes (medication ON/OFF). Real series violate all three
(see `STATISTICAL_VALIDATION.md` §S1: AR(1) ρ = 0.6 and t₃ noise inflate the 5% rate to 6–7.5%).

*Arbitrary constants.* `ZERO_VARIANCE_REL_TOL = 0.05`, `ZERO_VARIANCE_ABS_TOL = 1e-6`, `MIN_READINGS_FOR_STATS = 3`. All provisional.

### B. Quesenberry Q-statistics — `longitudinal.py::q_statistics`

For `i > m₀` (m₀ = 2), with `m = i − 1` prior readings:

```
r_i = sqrt( m/(m+1) ) · (x_i − x̄_{1..m}) / s_{1..m}   ~  t_{m−1}      (iid Gaussian, in control)
Q_i = Φ⁻¹( F_{t,m−1}(r_i) )                              ~  N(0,1), mutually independent
```

*Basis.* Quesenberry (1991). Verified against closed-form t CDFs for df = 1, 2, 4. The implementation takes the smaller tail to keep
precision, and caps `|Q|` at 8. Zero-variance history yields `None`. Affine invariance (property test) and finite/capped output
(property test) hold.

*Limitation inherent to the method.* A sustained shift is absorbed into the running mean, so the sensitivity to **old** shifts
decays (this is why `sequential_min_n` and the autocorrelation gate exist).

### C. CUSUM and EWMA — `longitudinal.py`

```
C⁺_t = max(0, C⁺_{t−1} + z_t − k)      C⁻_t = max(0, C⁻_{t−1} − z_t − k)      alarm: C > h          (k = 0.5, h = 5)
w_t  = λ z_t + (1−λ) w_{t−1}, w_0 = 0   limit_t = ±L · sqrt( λ/(2−λ) · (1 − (1−λ)^{2t}) )          (λ = 0.2, L = 3)
```

Hand-verified, including a property worth knowing: with λ = 0.2 and L = 3 the EWMA **can never alarm** on a sustained +1.0 SD
shift. Then `w_t = 1 − 0.8^t` and the limit is `sqrt(1 − 0.64^t)`; since `sqrt((1−a)(1+a)) ≥ 1−a` for `a = 0.8^t`, the path stays inside the
limit for every t (the asymptotic limit is exactly 1.0). Shifts above 1 SD can alarm; a 1 SD shift is detected by the CUSUM only. Properties tested: CUSUM sums are non-negative and alarm
exactly when a sum first exceeds `h`; with all `|z| ≤ k` neither sum ever leaves 0; the EWMA is a convex combination (bounded by
`max|z|`) and its limits widen monotonically to the asymptote `L·sqrt(λ/(2−λ))`.

*Operating characteristics were never verified in Phase 2; they are now measured* — see `STATISTICAL_VALIDATION.md` §S2: the
self-starting pair raises a family-wise false alarm in 10.3% (95% CI 9.4–11.3%) of 40-reading iid-Gaussian series, before the
autocorrelation gate.

*Arbitrary constants.* `k = 0.5`, `h = 5`, `λ = 0.2`, `L = 3` follow textbook SPC conventions (Montgomery) but are not tuned to
neurological data.

### D. Mann-Kendall — `longitudinal.py::mann_kendall`

```
S = Σ_{i<j} sgn(y_j − y_i)            Var(S) = [ n(n−1)(2n+5) − Σ_g t_g(t_g−1)(2t_g+5) ] / 18
z = (S − sgn S) / sqrt( Var(S) · infl )        p = erfc(|z|/√2)         infl = (1+ρ)/(1−ρ) if ρ > 0 else 1   (ρ ≤ 0.9)
```

Verified against hand-computed S and variance, including ties. Properties: invariant to strictly increasing transforms; negation
flips S and keeps p; a strictly monotone series attains `S = n(n−1)/2`; p ∈ [0,1].

**Incomplete.** The `(1+ρ)/(1−ρ)` factor is the large-sample variance inflation for an AR(1) *mean*; applying it to the
*Mann-Kendall S* is an approximation, and ρ is estimated from short detrended residuals (biased low). Measured effect
(`STATISTICAL_VALIDATION.md` §S3): at nominal α = 0.01 with AR(1) ρ = 0.6 the corrected test rejects 4.7% (n = 20) and 3.1%
(n = 40) of trend-free series; without the correction 12.5% and 16.9%. The correction helps a lot but is **not adequate** at strong
autocorrelation. The code already warns (`positive_autocorrelation`) when ρ > 0.3, but it still reports the p-value. A proper fix
(e.g. Hamed-Rao modified MK, or a block bootstrap) is a design change, not a bug fix, and was not made silently.

### E. Theil-Sen and MAD

`slope = median_{i<j} (y_j − y_i)/(t_j − t_i)` via scipy's implementation. Verified against a brute-force O(n²) median, and the
slope is equivariant (`y → a·y + b` scales it by `a`; shifting `t` leaves it unchanged — property test).
`σ̂ = 1.4826022185056018 · MAD` where the constant is `1/Φ⁻¹(0.75)`; falls back to the sample SD when MAD = 0, and reports
`zero_variance` when both are 0.

### F. `analyze_series`

Fuzzed with 3000 random series × 3 metrics: 0 exceptions, always JSON-serialisable with `allow_nan=False`. Deterministic
(property test). The set of signals is stable under affine rescaling, allowing at most one borderline method to flip.

### G/H. Change classification and polarity — `metric_semantics.py`

Provisional rule (no approved MCID):

```
Δ = current − baseline            δ = Δ / |baseline|       round(δ, 9) is compared (so 20% is not lost to 0.19999999999999996)
|δ| < 0.05  → stable
polarity known:   adverse = ±δ;  adverse ≥ 0.20 → deteriorating;  adverse ≤ −0.20 → improving;  else stable
polarity unknown: |δ| ≥ 0.20 → changed_direction_unknown;  else stable
baseline = 0: both 0 → stable; otherwise baseline_zero_undefined (no percentage invented)
```

Clinician-approved MCID rule (only if the entry has `approved_by` and `source`): classify on the absolute change with a tolerance
`1e-9·mcid + 8·ε·max(|b|,|c|)` (D02).

*Defects fixed.* D02 (a change equal to the MCID was "stable" for 1.1→1.4) and D03 (substring polarity matching).

*Still provisional / unresolved.* `STABLE_BAND = 0.05`, `ADVERSE_THRESHOLD = 0.20` are conventions, not clinical standards, and the
polarity registry is a conservative convention list. **Clinician approval required** for both, per metric. Relative change has no
near-zero-baseline guard (a baseline of 1e-9 produces enormous percentages); the zero-baseline case alone is handled.

### I. Risk heuristic — `agents/risk.py`

Exhaustive enumeration of all reachable inputs (`ml/simulation/p3_statistical_validation.py` S8):

```
fall_risk         = 0.25·[tremor_increase] + 0.40·[gait_slowing] + 0.25·[longitudinal_deteriorating] + 0.10·[missed_checkins ≥ 3]
seizure_risk      = min(1, 0.35·[low_adherence] + 0.10·min(recent_seizures, 4) + 0.30·[urgency_high])
deterioration     = 0.50·[longitudinal_deteriorating] + 0.25·[diagnostic_confidence > 0.5]
bucket(score)     = critical ≥ 0.75 > high ≥ 0.50 > medium ≥ 0.25 > low
overall           = max of the three sub-scores
```

Bucket boundaries, hand-summed scores, saturation, and monotonicity are verified by independent tests. **Every weight and cut
point is an unvalidated constant** with no clinical source. Structural issues documented rather than altered (changing the
weights would be inventing clinical thresholds): the same *deteriorating* flag raises both `fall_risk` (+0.25) and
`deterioration_risk` (+0.50), so sub-scores are not independent; `overall = max` lets one saturated factor dominate; the
diagnostic agent's retrieval *similarity* is added as if it measured clinical deterioration; a missing input lowers the score
exactly as a good value would (coverage is reported separately). **Status of the weights: unverified; clinician approval
required.** This score is not a probability and must not be presented as one.

### J. Evidence consensus — `core/consensus.py`

```
group(H, g)   = the strongest supporting item among those declared in independence group g      (duplicates count once)
support(H)    = Σ_g w_g · s_g / Σ_g w_g           w_g = reliability weight of the source agent, s_g ∈ [0,1]
contradiction = the strongest contradicting item
rank(H)       = max(0, support(H) − contradiction(H))
```

Hand example (test): strengths 0.6 (w = 0.65) and 0.8 (w = 0.90) in two groups → (0.39 + 0.72)/1.55 = **0.716**. Properties:
permutation invariance of agents and items, scores in [0,1], duplicating an item inside its group changes nothing. The outcome
carries `probability = None`, `calibrated = False`, `score_kind = "uncalibrated_support_index"`.

Honesty checks that pass: abstaining / skipped / failed agents contribute nothing and are not counted as agreement; conflicting
evidence reduces the rank score and is reported as disagreement; a tie has no leader (confidence 0).

**Incomplete (known limitation).** Independence is *declared by the emitter*, not verified. Two agents reading the same upstream
value but declaring different groups are counted as two sources. Simulation S7 shows this: the "shared-error" case reports 2
independent sources but is only as accurate as one source (65.3% vs 64.3% single vs 70.3% truly independent). A test pins the
behaviour so it cannot silently change. D09 removed the one in-repo case of double counting (the patient-history prior added to
retrieval confidence).

### L. Retrieval scoring

Lexical fallback: `score = |query tokens ∩ condition tokens| / |query tokens|` (query-token coverage, not Jaccard). D06: English stopwords alone scored 0.857 against
three conditions; stopwords are now removed (negations kept). The history prior (+0.15) is recorded separately and is *not* used as
evidence strength (D09). The semantic path needs Gemini embeddings and a Supabase pgvector database: **not executed in Phase 3**
(no credentials/network), so its similarity scores are **unverified**.

### M. DDI severity — `services/ddi.py`

Pair severity is the maximum over sources under the order `low < medium < high`; unknown labels never lose to a known one (rank 99,
D04). Unspecified labels rank 0. Names are NFKD-normalised, accent-folded, non-Latin scripts preserved (D05); unusable names raise
a `ddi_name_unusable` warning and `complete = False`. A curated non-exhaustive table can only say `combination_not_assessed`,
never "no interaction" (statement field enforces the wording). Properties: pair symmetry, duplicate/case insensitivity.
**The curated interaction table itself is an illustrative list and has not been clinically reviewed.**

### N. Safety gate — `core/safety.py`

Low-risk autonomy requires `disagreement ≤ 0.4`, `top_confidence ≥ 0.6`, `data_quality ≥ 0.3`. D01: any non-finite or missing value
now blocks autonomy (previously NaN passed because every comparison with NaN is false, and `None` raised a TypeError). D12: the
orchestrator substituted 0.5 for an *unknown* data quality, which passes the gate; it now substitutes 0.0 and logs
`data_quality_unknown_fail_closed`. Emergency escalation is unchanged.
Inconsistent constants noted, not changed: this gate uses 0.3; consensus uses `MIN_DATA_QUALITY_FOR_ACTION = 0.4`. When a consensus
leader exists, `disagreement ≤ 0.4` and `top_confidence ≥ 0.6` are the same condition. **All gate thresholds require clinician
approval.**

### O. Signal analysis — `agents/signal_analysis.py`

Tremor: flag if `value > 1.5·baseline` (baseline > 0). Gait speed: flag if `value < 0.8·baseline`. Arithmetic and percentage text
verified by inspection and tests; `1.5` and `0.8` are provisional and unvalidated. Confidence is `compared/evaluated` (a coverage
fraction, labelled as such), not a probability. Non-finite and boolean values are rejected.

### P. Adherence, completeness, data quality

D08: the adherence mean now counts only active medications. Not changed: the unweighted mean masks one very low-adherence drug among
several good ones; `compute_completeness` is an arbitrary 8-field fraction; `data_quality` conflates structural and numeric
invalidity. All three are **incomplete** and are listed in `DEFECTS_AND_FIXES.md` §Open observations.

### Q. Evaluation metrics — `ml/evalkit/metrics.py`

Verified against independent computations:
`Wilson: (p̂ + z²/2n ± z·sqrt(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)` (scipy and analytic agree); `ECE = Σ_b (n_b/n)|ȳ_b − p̄_b|`
(equal-width bins); `PPV = Se·π / (Se·π + (1−Sp)(1−π))`; F1; ROC-AUC against the Mann-Whitney U formulation; Brier; regression
metrics; cluster bootstrap resampling whole subjects. Properties: Wilson contains the estimate and shrinks with data; `AUC(1−p) = 1−AUC(p)`,
AUC invariant under monotone transforms; ECE and Brier ∈ [0,1].

### R. OOD detector — `app/ml/ood.py`

Layers: identity, range (training range ± 10% margin), ordering, and Mahalanobis distance with a Ledoit-Wolf shrinkage covariance,
threshold at the 0.99 quantile of held-out (leave-group-out) squared distances; `z_limit = 8` is only a per-feature fallback when the
training min/max are unavailable. Formulas verified against a
numpy reference. **Detection power is limited** (S6): false rejection ≈ 1.3% (target ≤ 1%+range), but a +3 SD shift in one
feature goes undetected 47% of the time and a shift of every feature by 1 SD 75%; breaking a correlation by shuffling a pair is
missed 98% of the time. Gross unit errors (×1000) are always caught (0% missed). The detector is a coarse guard, not a
distribution-shift monitor.

### S. Trained models

`ml/models/parkinsons_risk_classifier_metadata.json`: logistic regression selected over a random forest by `GroupKFold`, pooled
ROC-AUC **0.640** over **42 subjects**, accuracy 0.630, F1 0.553. Selection and the reported figure use the same folds, so
the number is optimistic (S4 shows best-of-7 selection on null data yields ~0.65). `class_weight='balanced'` scores are
approximately posteriors under a **50/50 prior**, not probabilities for the deployment population (S5; D10 labels them
uncalibrated). **Real-data re-validation was attempted and is blocked** (UCI downloads return proxy 403), so no new real-data
number exists for Phase 3.

---

## 3. Unsupported assumptions (consolidated)

1. Within-patient readings are iid Gaussian (A–D). Real series are autocorrelated, heavy-tailed and state-dependent.
2. Agents' declared independence groups are truthful (J).
3. Every threshold in A–J, N, O is a provisional engineering value, not a clinical one.
4. Retrieval similarity ≈ clinical relevance (L); the curated knowledge base is complete and correct (unreviewed).
5. The curated interaction table is representative (M).
6. Training data (UCI Parkinsons Telemonitoring, voice features) is representative of deployment inputs — it is a single-source,
   42-subject research data set (S).

## 4. Values requiring clinician approval — UNRESOLVED

`STABLE_BAND` 0.05 · `ADVERSE_THRESHOLD` 0.20 · per-metric MCIDs and polarity · risk weights and bucket cut points ·
tremor 1.5× and gait 0.8× · safety gate 0.4 / 0.6 / 0.3 · `MIN_DATA_QUALITY_FOR_ACTION` 0.4 · missed-check-in count (3) and
recent-seizure cap (4) · diagnostic-confidence 0.5 used in the deterioration score · interaction severities · completeness fields.
None of these were changed in Phase 3.
