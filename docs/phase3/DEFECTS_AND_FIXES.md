# NEURO-MAS Phase 3: Defects and Fixes

Each defect below was **reproduced by a failing test before any code was changed**, then fixed, and the test now passes. The
regression suite for D01–D11 was run against the unfixed code first: **67 tests, 41 failed, 26 passed** (the 26 passes are
control cases that pin the behaviour that must *not* change). The D12 test was written afterwards in the same way (1 failed, 1 control
passed). Output files are in the session scratchpad (`defects_before_fix.txt`).

Severity is my own assessment of the *worst plausible effect inside this prototype*; none of these have been observed to harm anyone,
and the system has never been used with real patients.

Interfaces were preserved. The only API-visible change is two **additive** fields (D10).

| ID | Sev. | Component | One-line defect | Status |
|---|---|---|---|---|
| D01 | High | `core/safety.py` | NaN / ±inf gate inputs **permitted autonomous action**; `None` raised `TypeError` | Fixed |
| D02 | Medium | `core/metric_semantics.py` | A change exactly equal to the clinician-approved MCID classed "stable" (float error) | Fixed |
| D03 | High | `core/metric_semantics.py` | Substring polarity matching assigned the **wrong direction** to metrics | Fixed |
| D04 | High | `services/ddi.py` | Unrecognised severity labels (e.g. "contraindicated") were outranked by "low" | Fixed |
| D05 | Medium | `services/ddi.py`, `clinical_rules.py` | Non-Latin / accented drug names silently dropped from the check | Fixed |
| D06 | Medium | `services/knowledge_base.py`, `rag/knowledge_retriever.py` | Stopwords alone matched conditions (score 0.857) | Fixed |
| D07 | Medium | `ml/personalized_anomaly.py` | Float-noise SD produced a z-score of 3.6e9 | Fixed |
| D08 | Medium | `services/repository.py` | Inactive medications dragged the adherence mean down | Fixed |
| D09 | Medium | `agents/diagnostic.py`, retriever | The patient-history prior (+0.15) was counted as symptom evidence in consensus | Fixed |
| D10 | Medium | `schemas/ml_demo.py` | Class-weighted scores returned as `probability` with no calibration flag | Fixed (labelled) |
| D11 | Low | `agents/personalization.py` | Fixed 0.7 "confidence" looked like a measurement | Fixed (labelled) |
| D12 | High | `orchestrator/orchestrator.py` | **Unknown data quality was replaced by 0.5**, which passes the 0.3 safety gate | Fixed |

---

## D01 — Non-finite safety-gate inputs fail open

*Evidence before.* `SafetyDecision(permitted_autonomously=True, ..., reasons=['Confidence, data quality and disagreement all within
bounds for a low-risk action.'])` for `disagreement = NaN`, `top_confidence = NaN`, `data_quality = NaN` and ±inf cases. With
`None`: `TypeError: '>' not supported between instances of 'NoneType' and 'float'` at `safety.py:117/122/126`.
*Cause.* Every ordered comparison with NaN is `False`, so none of the "block" conditions fired.
*Fix.* `_is_finite_number()` (rejects bool, None, NaN, ±inf, non-numbers). In the `EXECUTE_LOW_RISK` rule, before any threshold
comparison, a non-finite input returns `permitted_autonomously=False, requires_human_approval=True` with the reason
"Gate input(s) [...] are not a finite number (non-finite or missing)...". Emergency escalation is unaffected.
*Behaviour change.* Previously-permitted NaN cases now require approval; `None` no longer raises.
*Tests.* `test_p3_defect_regressions.py` D01 group (parametrised over NaN/inf/-inf/None for each input, plus controls with finite values).

## D02 — MCID boundary lost to floating-point error

*Evidence before.* `classify_change` with an approved MCID of 0.3: `(1.1 → 1.4)` Δ = 0.2999999999999998 → `'stable'` (expected
`'deteriorating'`); the same for `(7.7 → 8.0)`, and the mirror cases `(3.3 → 3.0)`, `(1.4 → 1.1)` → `'stable'` instead of `'improving'`.
*Fix.* Compare with a tolerance `1e-9·mcid + 8·ε·max(|baseline|, |current|)`; a change equal to the MCID up to rounding counts.
*Behaviour change.* Only changes within ~1e-9 relative of the MCID; clearly smaller changes still classify as stable (controls).
*Note.* The MCID itself still has to be clinician-approved (`approved_by`, `source`); no value is shipped.

## D03 — Metric polarity from substrings

*Evidence before.* `stride_time_variability` → `lower_worse` (it contains "stride"), `missteps_count` → `lower_worse` (contains
"steps"), `seizure_free_days` → `higher_worse` (contains "seizure"). A rise in seizure-free days was thus "deteriorating".
*Fix.* Whole-token / adjacent-phrase matching; names containing a direction-blurring modifier (variability, variance, cv, sd, std,
asymmetry, free, impairment, complaint(s), loss, decline, improvement, reduction, change, delta) return `unknown`; a name matching
both families returns `unknown`. `unknown` is always the safe answer: the change is reported for a clinician rather than guessed.
*Behaviour change (conservative).* Some previously classified names are now `unknown`, so their changes are reported as
`changed_direction_unknown`. The curated polarity convention list still needs clinician approval.
*Tests.* D03 group, including case/separator insensitivity (also a property test).

## D04 — DDI severity ordering

*Evidence before.* Combining a source that says `contraindicated` with one that says `low` returned `low`.
*Fix.* `severity_rank`: low < medium < high (case-insensitive); unspecified (`""`, unknown, none, unspecified) = 0; any
**other** label = 99, never outranked. `_combine` uses it.
*Behaviour change.* A pair with a strong unrecognised label now reports that label rather than a lower one. Whether
such labels should be mapped onto the three-level scale is a clinical decision, left open.

## D05 — Drug names dropped

*Evidence before.* A non-Latin drug name (e.g. the Devanagari name for levodopa) normalised to an empty string and vanished from the
pair list with no warning; accented spellings were not matched to the curated table.
*Fix.* NFKD + accent stripping for Latin text, NFKC casefold for non-Latin scripts; `fold_accents()` in `canonical_drug_names`;
names that are still unusable are reported as a `ddi_name_unusable` warning and set `complete = False`.
*Behaviour change.* Previously invisible names are now visible and the report is marked incomplete. Matching a non-Latin name to
a drug still requires a synonym in the source; none were invented.

## D06 — Stopwords scored as symptom matches

*Evidence before.* A query consisting only of English function words scored **0.857** against Alzheimer's disease, essential
tremor and meningitis.
*Fix.* `STOPWORDS` (grammar words only; negations are kept) removed from both the index and the query tokens, in the lexical
scorer and in the retriever's fallback.
*Behaviour change.* Lexical scores for ordinary queries shift slightly because the denominators lose function words; meaningful
overlaps are unchanged.

## D07 — Float-noise standard deviation

*Evidence before.* A constant series with an SD of 2.8e-17 gave `z = 3.6e9` and an "anomaly".
*Fix.* `SD_NOISE_FLOOR_REL = 1e-12`: if `s ≤ 1e-12·max(1, |mean|)` the zero-variance branch applies.
*Behaviour change.* None for real variation (the floor is 12 orders of magnitude below any plausible measurement spread).

## D08 — Adherence averaged over inactive medications

*Evidence before.* A discontinued medication with 0% recorded adherence lowered the patient's reported adherence, and could trip the
`< 0.8` seizure-risk condition.
*Fix.* Only medications with `active` true enter the mean.
*Not changed (open).* The mean is still unweighted, so one badly adhered drug among several good ones is masked (§Open).

## D09 — History prior counted as evidence

*Evidence before.* The retriever added +0.15 to the confidence of conditions already in the patient's history, and that inflated figure
became `HypothesisEvidence.strength` in the consensus: a record of past diagnosis was treated as present-symptom evidence.
*Fix.* The retriever records `retrieval_confidence` (before the prior) and `history_prior_applied`; the diagnostic agent uses
`retrieval_confidence` as the evidence strength and notes `history_prior_excluded_from_strength` in provenance. The displayed
candidate `confidence` is unchanged.
*Behaviour change.* Consensus support index for a history-matched condition is lower by up to the prior.

## D10 — Uncalibrated scores presented as probabilities

*Evidence.* Simulation S5: `class_weight='balanced'` yields ECE 0.20–0.36 for deployment prevalences ≠ 50%.
*Fix (labelling, not recalibration; the correction needs the unknown deployment prevalence).* `ParkinsonsPredictionOut` and
`ParkinsonsRiskPredictionOut` gain `probability_calibrated: bool = False` and `probability_note: str = UNCALIBRATED_NOTE`.
*Behaviour change.* Two additive response fields. The frontend ml-lab page does not yet display the note (not done).

## D11 — Placeholder confidence

*Fix.* `personalization.py` adds the warning `confidence_is_a_fixed_placeholder_not_a_measure`; the value stays 0.7 so downstream code
and tests are undisturbed.

## D12 — Unknown data quality treated as adequate (found late in Phase 3)

*Evidence before.* Orchestrator with the intake agent failing (`RuntimeError`) and a high-confidence leader present:
the low-risk action had `requires_human_approval = False` (test failed on
`assert acts and all(a.requires_human_approval is True for a in acts)`).
*Cause.* `data_quality = data_quality_q if data_quality_q is not None else 0.5`; 0.5 is above the 0.3 gate, so a failed or abstaining
intake step left the system acting on data nobody had assessed.
*Fix.* Unknown or non-numeric data quality → `0.0` (fail closed) plus an audit event `data_quality_unknown_fail_closed` recording the
intake status. Emergency escalation is unaffected.
*Behaviour change.* When the intake agent does not complete, low-risk actions require human approval where before they ran.
Full suite after the change: 689 passed, 26 skipped (no regression).
*Tests.* `test_D12_control_good_data_quality_allows_the_low_risk_action` (guards against over-blocking) and
`test_D12_unknown_data_quality_is_not_assumed_adequate`.

---

## Findings that were checked and are **not** defects

* t-table / Cornish-Fisher critical values (≤ 4.7e-4 and < 1e-4 against scipy); Q-statistics against closed-form t CDFs; Mann-Kendall
  with ties; Theil-Sen against brute force; MAD constant; Wilson / ECE / PPV / AUC / Brier against independent computations.
* `analyze_series` fuzz: 3000 random series × 3 metrics, 0 exceptions, JSON always finite.
* Single-review false-positive rate of the prediction interval attains 5% (S1; the n = 30 cell excursion was chance, 2M-trial recheck).

## Open observations — documented, deliberately **not** changed

These involve clinical choices or design decisions, not demonstrable arithmetic faults.

1. **Mann-Kendall autocorrelation correction is inadequate at ρ = 0.6** (S3: 3–5% false trends at nominal 1%). A modified MK
   (Hamed-Rao) or block bootstrap would be a design change.
2. **Self-starting charts: ~10% family-wise false alarms per 40 readings** before the autocorrelation gate (S2).
3. **Consensus independence is declared, not verified** (S7); a pinned known-limitation test exists.
4. **OOD detector power is weak** for moderate and correlation-structure shifts (S6).
5. Unweighted mean adherence masks a single low-adherence drug.
6. `compute_completeness` is an arbitrary 8-field fraction; `data_quality` conflates structural and numeric invalidity.
7. Safety `data_quality < 0.3` vs consensus `MIN_DATA_QUALITY_FOR_ACTION = 0.4` are inconsistent constants; the `disagreement ≤ 0.4`
   and `top_confidence ≥ 0.6` gates are the same condition when a consensus exists.
8. `classify_change` has no near-zero-baseline guard (only exact zero).
9. The risk heuristic's flag reuse (`deteriorating` in two sub-scores), `max` aggregation, and diagnostic-similarity-as-deterioration term.
10. Legacy `compute_consensus` is unused in app code.
11. `rejected_fixed_baseline` numbers in `anomaly_study.py` are hard-coded literals (S2 recomputes the equivalent from code).
12. Shipped risk classifier: accuracy 0.630 vs 0.618 majority baseline; no CI; selection on evaluation folds; no time-only baseline.

---

## Commands run and results (exact)

All commands were run in this session. The Python used is the venv at
`/tmp/claude-0/-home-claude/d4976552-b1de-5662-add6-151463a8a443/scratchpad/venv/bin/python` (Python 3.13.17).

```
# Backend, before any Phase 3 change (baseline)
cd backend && python -m pytest tests -q -p no:cacheprovider
  -> 551 passed, 26 skipped, 36 warnings in 76.22s

# D01–D11 regression tests against UNFIXED code
python -m pytest tests/phase3/test_p3_defect_regressions.py -q -p no:cacheprovider
  -> 41 failed, 26 passed in 2.84s            (26 passes are control cases)

# Backend after D01–D11
python -m pytest tests -q -p no:cacheprovider
  -> 687 passed, 26 skipped, 36 warnings in 64.78s

# D12 test, before and after the orchestrator change
python -m pytest tests/phase3/test_p3_defect_regressions.py -q -p no:cacheprovider -k D12
  -> before: 1 failed, 1 passed          after: 2 passed

# Property tests (hypothesis 6.170.8, derandomize=True)
python -m pytest tests/phase3/test_p3_properties.py -q -p no:cacheprovider
  -> 25 passed in 14.22s

# All Phase 3 tests
python -m pytest tests/phase3 -q -p no:cacheprovider
  -> 163 passed in 17.43s
     (reference_math 30 · reference_scoring 39 · defect_regressions 69 · properties 25)

# FINAL backend run
python -m pytest tests -q -p no:cacheprovider
  -> 714 passed, 26 skipped, 36 warnings in 73.48s

# Statistical validation and large PI check
PYTHONPATH=. python -m ml.simulation.p3_statistical_validation --out ml/simulation/results/p3_statistical_validation.json
  -> exit 0, real 2m11.995s
PYTHONPATH=. python -m ml.simulation.p3_pi_large_check
  -> see STATISTICAL_VALIDATION.md S1 (false-positive rates 0.0496–0.0500)

# Frontend
cd frontend
npm run typecheck   -> tsc --noEmit, exit 0
npm test            -> Test Files 4 passed (4), Tests 24 passed (24), exit 0
npm run build       -> exit 0
npm run e2e         -> 8/8 checks passed, exit 0   (mocked-API smoke against the production build)
```

Skipped tests (26 = 22 + 4): 22 live Row-Level-Security tests (`tests/rls_live/test_rls_matrix.py`; they need a separate Supabase test project and
four `NEUROMAS_*` credentials) and 4 live Supabase/Gemini API tests (`tests/test_api.py`; `NEUROMAS_LIVE_TESTS` not set). **None of the live
integrations were executed.**

### Do the new tests actually detect faults? (mutation spot-check)

Three faults were planted into the code under test and the property tests rerun:
* CUSUM without the `max(0, ·)` reset → **caught** (`test_cusum_paths_are_nonnegative_and_alarm_iff_boundary_crossed`);
* prediction-interval z-score without dividing by the SD → **caught** (`test_point_detector_affine_invariance`);
* Wilson interval replaced by the Wald interval → **not caught by the properties** (Wald also satisfies containment and shrinkage), but caught by the
  exact scipy/analytic Wilson reference tests in `test_p3_reference_scoring.py`. The property and reference suites are complementary.

### Not done

* No real-data model validation (UCI downloads blocked with 403).
* No live Gemini / Supabase / pgvector execution; semantic retrieval scores are unverified.
* The frontend does not yet show `probability_note`.
* Nothing is pushed: the remote rejects pushes from this environment (403), so the branch `phase2/advanced-ml-hardening` carries
  12 unpushed local commits, plus the Phase 3 commit once made.
