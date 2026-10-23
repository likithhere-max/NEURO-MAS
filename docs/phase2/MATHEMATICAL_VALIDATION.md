# Mathematical Validation (Phase 2, C)

Scope: what each numerical method does, how its implementation was checked, and — separately — what is **not** known.
"Verified" below means *the arithmetic matches an independent hand calculation, a reference library, or a simulation*.
It never means clinically valid. Every threshold is a provisional engineering parameter.

Reproduce everything in this file with the commands in RUN_AND_REPRODUCTION.md.

## 1. Inventory

| Component | File | What it computes | Implementation check | Statistical-behaviour check | Clinical validity |
|---|---|---|---|---|---|
| Metric polarity + change classification | `app/core/metric_semantics.py` | relative/absolute change; direction-aware label | `test_metric_semantics.py`, `test_metric_thresholds.py`, properties in `test_math_properties.py` | none (convention) | **None.** 5 % / 20 % are placeholders. |
| Point anomaly (Student-t prediction interval) | `app/ml/personalized_anomaly.py` | does the newest reading fall outside the t prediction interval of the prior readings | hand/reference tests (`test_personalized_anomaly.py`) | repeated-review study (§3) | None |
| Robust z (median/MAD) | `app/ml/longitudinal.py` | report-only context | hand tests | — | None |
| Theil–Sen slope (+CI) | `app/ml/longitudinal.py` | median of pairwise slopes | **matches `scipy.stats.theilslopes`**, hand example, outlier test, irregular times | — | None |
| Mann–Kendall (+ties, +lag-1 variance correction) | same | monotone-trend test | hand examples incl. tie correction; sign symmetry | simulation (§3) | None |
| Quesenberry Q (self-starting standardisation) | same | z-like statistic of each reading vs. its own history | hand example; **standard-normal & uncorrelated under null** test; location-scale invariance property | simulation (§3) | None |
| Tabular CUSUM (k = 0.5, h = 5) / EWMA (λ = 0.2, L = 3) on Q | same | sequential shift detection | hand examples | simulation (§3) | None |
| OOD / abstention | `app/ml/ood.py` | contract, physical bounds, algebraic identities, marginal range, optional Mahalanobis | `test_ood.py` (13 tests incl. properties) | **not measured** on real data (no data) | None |
| Evidence consensus | `app/core/consensus.py` | ordering of hypotheses | `test_consensus_evidence.py`, `test_math_properties.py` | — | None; **not a probability** (CONSENSUS_DESIGN.md) |
| Evaluation kit (metrics, CV, bootstrap, permutation) | `ml/evalkit/*` | model-evaluation statistics | reference values, leakage guards (`test_ml_evalkit.py`, 30+ tests) | — | n/a |
| Risk score (agent) | `app/agents/risk.py` | weighted additive heuristic | unit tests | none | **Stays a heuristic.** Its output is labelled as such; it is not a calibrated risk. |

## 2. Principles applied
1. **Undefined stays undefined.** Single-class AUC, zero-variance history, zero baseline, too few readings: the function
   returns `None` / `not_assessed` / `baseline_zero_undefined` with a reason, never `0`, `1`, or an imputed value
   (`test_single_class_metrics_are_none_not_zero`, `test_q_statistics_zero_variance_history_is_undefined_not_infinite`,
   `test_insufficient_and_invalid_inputs_are_labelled_not_guessed`).
2. **Methods are chosen by what the data supports.** Methods that need ≥ 8 (trend) or ≥ 15 (sequential) readings are not run
   on shorter series and appear as `not_assessed`.
3. **Thresholds carry provenance** (`AnalysisConfig.*_provenance`, `threshold_basis` in every classification) and are
   overridable. An MCID is applied only if it is in the configured file **with `approved_by` and `source`**; otherwise it
   is ignored and the output says so (`test_unapproved_mcid_is_ignored_with_a_note`).
4. **A signal is not a diagnosis.** Reports carry `clinical_significance.assessed = false` unless an MCID is configured.

## 3. Simulation evidence for the longitudinal detector

Files: `ml/simulation/anomaly_study.py` → `results/anomaly_study.json` (seed 20261023, 400 replications per scenario,
series of 40 readings, Python 3.13.17, NumPy 2.5.4) and `ml/simulation/point_test_repeated_review.py` →
`results/point_test_repeated_review.json` (1000 replications). Monte-Carlo standard error for a rate near 7 % with 400
replications is about ±1.3 percentage points; treat differences smaller than that as noise.

### 3.1 Why the point test's SIGNAL level is α = 0.001
The point test is re-applied at every new reading. Null (iid Gaussian, no change), 31 reviews at n = 10…40:

| single-test α | false alarm at one review (n = 20) | false alarm at **any** of 31 reviews |
|---|---|---|
| 0.05 | 6.7 % | **79.6 %** |
| 0.01 | 1.8 % | 25.6 % |
| 0.001 | 0.1 % | 2.1 % |

So α = 0.05 is reported only as an informational note; the *signal* level is 0.001.

### 3.2 False-alarm rates under the null with the production configuration (autocorrelation gate ρ ≤ 0.3)
"any" = any of point / trend / CUSUM / EWMA signal. FW = at least one signal across reviews at n = 10…40.

| Null scenario | any, one review at n = 20 | any, one review at n = 40 | FW any (n = 10…40) |
|---|---|---|---|
| iid Gaussian | 1.8 % | 6.8 % | 11.5 % |
| AR(1) ρ = 0.3 | 8.3 % | 19.0 % | 29.5 % |
| AR(1) ρ = 0.6 | 12.2 % | 19.0 % | 37.0 % |
| iid Student-t (3 d.f., heavy tails) | 7.5 % | 20.3 % | 43.0 % |

**Interpretation.** With white Gaussian noise the combined rule has ≈ 7 % false alarms at one review of 40 readings and
≈ 12 % if the series is reviewed continuously. With realistic autocorrelation or heavy tails the rule fires on pure noise in
**30–43 % of pure-noise series** (FW column) over a course of monitoring. That is a property of the method, reported rather
than hidden: signals must be read as "a clinician may want to look", and the rate is not acceptable for any automatic
action. Real physiological series are typically autocorrelated, so the iid row is the optimistic one.

### 3.3 Effect of the autocorrelation gate (AR(1) ρ = 0.6, n = 40, any signal)

| Gate (suppress CUSUM/EWMA if pre-alarm lag-1 ρ above) | false alarm (one review, n = 40) |
|---|---|
| off | 63.5 % |
| 0.3 (**production default**) | 19.0 % |
| 0.2 | 9.0 % |

The gate removes most—not all—of the inflation; the 0.2 setting is stricter but costs power (§3.4).

### 3.4 Power (alternatives; 40 readings, shift/drift starting mid-series, gate 0.3)

| Alternative | any signal | CUSUM | EWMA | trend | point (α = 0.001) |
|---|---|---|---|---|---|
| step of 2 SD | 96.8 % | 90.0 % | 80.5 % | 80.0 % | 0.3 % |
| step of 3 SD | 95.5 % | 86.0 % | 83.3 % | 83.8 % | 0 % |
| drift totalling 3 SD | 99.8 % | 78.3 % | 72.8 % | 99.0 % | 0.3 % |

The α = 0.001 point test has essentially no power against sustained shifts; that is by design (it is a gross-outlier
alarm), and sustained change is the job of CUSUM/EWMA/trend.

### 3.5 Limits of this evidence
* Simulated noise only (iid Gaussian, AR(1), Student-t₃). No real patient series was analysed.
* Shifts and drifts were injected at one size/time; no study of missing data, irregular sampling, seasonality, or
  state-dependent (e.g. on/off medication) variation.
* The "fixed-baseline CUSUM/EWMA" figures in the `rejected_fixed_baseline` block of `anomaly_study.json` (30–65 % over 35
  steps for 5–12 baseline readings) are **constants copied from an earlier exploratory 20,000-replication run; the
  committed script does not regenerate them**. Treat that block as a recorded observation, not a reproducible result.
  The 400-replication run in the same file shows the self-starting charts at 6.8 % (CUSUM, FW, iid, gate off) / 5.3 %
  (EWMA), consistent with, but not identical to, the 6.9 % / 7.0 % quoted in that block.
* Default parameters (k, h, λ, L, 15-reading minimum, ρ gate) were chosen by looking at these same simulations, so the
  reported null rates are tuned rates, not out-of-sample.

## 4. OOD / abstention (what is and is not established)
Established by tests: the training-mean vector is accepted for both shipped models; unit errors (e.g. Hz vs kHz),
negative jitter, broken `Flo ≤ Fo ≤ Fhi` ordering, broken `DDP = 3·RAP` / `DDA = 3·APQ3` identities, percent-vs-fraction
mixups and non-finite values produce an **abstention** (no prediction, no probability) rather than a number; a JSON domain
serialises losslessly with no pickle.
Not established: false-rejection and missed-OOD rates on real data. The shipped domain is derived from each model's own
`StandardScaler` moments (`scaler_moments_provisional`), which has no min/max or covariance. A fitted domain
(`fit_domain`) exists but was **not** produced from real data (dataset unreachable). Gaussian-style distances are not
valid density estimates for heavy-tailed acoustic features.

## 5. Things deliberately not done
* No probability is attached to the consensus, the risk heuristic, or any anomaly signal.
* No MCID is assumed for any metric.
* The ML demo models were not re-trained or re-measured (see ML_VALIDATION_RESULTS.md).
* No multiple-comparison correction across the several metrics a patient may have; each metric is analysed separately, so the
  per-patient false-alarm rate grows with the number of tracked metrics. (Open — OUTSTANDING_ISSUES.md.)
