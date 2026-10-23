# Phase 6: Final Readiness Report

**Verdict.** Phase 6 produced no new performance evidence, because the two public datasets remain unobtainable here (HTTP 403, folder access
denied, no authorised mirror). It produced what was possible without them: a tested, fail-closed evidence pipeline, a prespecified and
frozen extension protocol, machine-readable experiment and claims registers, planning arithmetic, an external-validation protocol, a hazard
log and an architecture audit. Every real-data experiment is `NOT_RUN`. **No clinical validity is claimed; no model, metric or threshold is
presented as clinically acceptable.** A 10/10 score is not offered because none can be supported; a reproducible `NOT_RUN` with a concrete
dependency list is the honest outcome.

## 1. What changed (all committed locally; not pushed)

| Area | Change |
|---|---|
| Metrics | `ml/evalkit/clinical_metrics.py`: 12 metrics, calibration intercept/slope, net-benefit curves, paired cluster-bootstrap differences, subgroup gates, exploratory cross-fitted recalibration |
| Extension | protocol `p6-1` (frozen, SHA-256 `7fd315bb10c38478c816c2bd552e8d799e0e36915ce027a50f49ab75b0606ec8`, additive to `p4a-1`), candidates with written hypotheses, `workflow --candidate-set p6-1-ext` |
| Report | `ml/validation/phase6_report.py`: recomputes metrics from saved held-out predictions, refuses mismatched folds, never upgrades an evidence label, fails closed |
| Provenance | contract 1.1.0 adds intended use, split identifiers, performance link; training can link a completed validation run |
| Planning | `sample_size.py`, `power_planning --sample-size` |
| Warnings | global import-time suppression removed (F7); counted warnings recorded in results |
| CI | `.github/workflows/backend-ci.yml` (never executed remotely) |
| Registers | `experiment_registry.py` + `experiment_registry.json`; `claims_register.py` + `claims_register.json` |
| Docs | `docs/phase6/*` (this directory) |

## 2. What ran

| Run | Result |
|---|---|
| Registered workflow, extension workflow, report pipeline (commands in `BASELINE_REPORT.md`) | each exit 2, `NOT_RUN`, committed at a clean commit |
| Voice sample-size simulation | completed; `PLANNING_ESTIMATE` |
| Full backend regression | **944 passed, 26 skipped, 83 warnings** (382.6 s), `python -m pytest -q -p no:cacheprovider -rs` from `backend/` |

Experiment registry (`backend/ml/validation/experiment_registry.json`; class derived from each output file, SHA-256 of every output checked by test):

| Experiment | Phase | What | Status | Class |
|---|---|---|---|---|
| `E-P2-FETCH` | 2 | Download of UCI 174 / 189 through the session proxy | `failed` | `BLOCKED` |
| `E-P2-ARTIFACTS` | 2 | Inspection of the shipped model artifacts (recorded metadata) | `completed` | `METADATA_CONSISTENCY_ONLY` |
| `E-P4A-ARTIFACT-AUDIT` | 4A | No-data audit of shipped artifacts (historical figures vs recorded counts) | `completed` | `METADATA_CONSISTENCY_ONLY` |
| `E-P4A-SELFTEST` | 4A | Harness integrity self-test on synthetic scenarios | `completed` | `SYNTHETIC_ENGINEERING_TEST` |
| `E-P4A-DISEASE` | 4A | Registered p4a-1 disease validation | `not_run` | `NOT_RUN` |
| `E-P4A-RISK` | 4A | Registered p4a-1 risk validation | `not_run` | `NOT_RUN` |
| `E-P5-POWER` | 5 | Design-stage precision and power planning | `completed` | `PLANNING_ESTIMATE` |
| `E-P5-WORKFLOW` | 5 | Registered workflow, no data | `not_run` | `NOT_RUN` |
| `E-P6-BASELINE` | 6 | Registered workflow p4a-1, no data (Phase 6 re-run) | `not_run` | `NOT_RUN` |
| `E-P6-EXTENSION` | 6 | Prespecified extension p6-1-ext, no data | `not_run` | `NOT_RUN` |
| `E-P6-REPORT` | 6 | Phase 6 report pipeline, no data | `not_run` | `NOT_RUN` |
| `E-P6-VOICE-SAMPLE-SIZE` | 6 | Voice-value sample-size simulation | `completed` | `PLANNING_ESTIMATE` |

## 3. What was reproduced

**Nothing from real data.** The historical figures remain `HISTORICAL_UNVERIFIED_NOT_REPRODUCED`. What was reproduced is engineering behaviour on forged or
fixture data (labelled `SYNTHETIC_ENGINEERING_TEST`) and arithmetic (majority baselines, sample sizes).

## 4. Claims (`backend/ml/validation/claims_register.json`; checked by `claims_register.py` and tests)

No claim is `VERIFIED_INTERNAL_EVIDENCE` or `VERIFIED_EXTERNAL_EVIDENCE`.

| ID | Claim | Class | Note |
|---|---|---|---|
| CL01 | The historical disease figures (LOSO accuracy 0.7744, AUC 0.7941) are reproduced from the public data. | `NOT_RUN` | Not reproduced: the data file is unavailable (HTTP 403). The figures stay HISTORICAL_UNVERIFIED_NOT_REPRODUCED. |
| CL02 | The historical risk figures (accuracy 0.6304, AUC 0.6405) are reproduced from the public data. | `NOT_RUN` | Not reproduced: the data file is unavailable. |
| CL03 | The shipped artifacts' recorded counts and metrics are internally consistent (integer confusion matrices, class counts equal scaler row counts). | `METADATA_CONSISTENCY_ONLY` | Consistency of recorded numbers only; says nothing about whether a leakage-free evaluation produced them. |
| CL04 | The majority-class baselines are 0.7538 (disease) and 0.6182 (risk), arithmetic on the recorded class counts. | `METADATA_CONSISTENCY_ONLY` | Arithmetic on recorded counts. The brief's 0.6178 has no traceable source. |
| CL05 | The brief's risk majority baseline of 0.6178 is supported by the repository. | `FAILED` | Claim not supported: the figure appears nowhere; 0.6182 is the traceable value. |
| CL06 | Voice features add measurable information beyond time and baseline UPDRS for the risk target. | `NOT_RUN` | Unknown. Requires the registered run; the result would be INCONCLUSIVE unless the p4a-1 rule is met. |
| CL07 | A voice gain of about +0.03 AUC is probably not resolvable with 42 patients; about +0.09 probably is (single-reference simulation, 12 repetitions per cell). | `PLANNING_ESTIMATE` | Optimistic planning figure with large Monte-Carlo error; not a result about the real data. |
| CL08 | With 24 PD and 8 healthy subjects the minimum detectable disease AUC at 80% power is about 0.835. | `PLANNING_ESTIMATE` | Analytic design-stage arithmetic. |
| CL09 | An external validation with half-width 0.05 on sensitivity and specificity needs about 784 subjects at 25% prevalence (assumed sensitivity 0.85, specificity 0.80). | `PLANNING_ESTIMATE` | Arithmetic with placeholder assumptions. |
| CL10 | Either model has been validated on an independent external cohort. | `BLOCKED` | No cohort available; protocol drafted; blocked on data access and clinical decisions. |
| CL11 | Either model is calibrated. | `NOT_RUN` | Calibration metrics are implemented but unmeasured. |
| CL12 | Any prespecified extension candidate improves on the registered baselines. | `NOT_RUN` | Extension never run; any future result is EXPLORATORY by protocol p6-1. |
| CL13 | Either model has clinical utility or may be used in patient care. | `BLOCKED` | No clinical claim is made. Intended use, endpoints and acceptance criteria need qualified clinical input. |
| CL14 | Dataset SHA-256 hashes are pinned in the dataset manifest. | `BLOCKED` | No file retrieved, so no hash can be honestly recorded. |
| CL15 | The shipped model artifacts carry COMPLETE training provenance. | `FAILED` | They are INCOMPLETE_DECLARED and cannot be backfilled truthfully; retraining on authenticated data would produce COMPLETE records. |
| CL16 | The registered protocol p4a-1 is unchanged since it was frozen, and p6-1 is additive. | `SYNTHETIC_ENGINEERING_TEST` | Verified by hash-pinning tests; an engineering fact about the repository, not performance evidence. |
| CL17 | The Phase 6 report pipeline fails closed (NOT_RUN, no numbers) when predictions or runs are missing, and refuses mismatched folds. | `SYNTHETIC_ENGINEERING_TEST` | Tested on forged data matched to stand-in artifacts. |
| CL18 | The metric library agrees with scikit-learn and hand calculations. | `SYNTHETIC_ENGINEERING_TEST` | Unit tests; forged inputs. |
| CL19 | Validation modules no longer install a global warning filter and run results record warning counts. | `SYNTHETIC_ENGINEERING_TEST` | Fixes finding F7. |
| CL20 | The CI workflow runs the tests and the real-data validation on GitHub. | `NOT_RUN` | Workflow written and YAML-parsed locally; never executed on GitHub because the push is not authorised. |
| CL21 | Phase 6 commits have been pushed to the remote. | `BLOCKED` | Push returns 403 and is not authorised; commits exist locally only. |

## 5. Defects and findings

| ID | Finding | Resolution |
|---|---|---|
| F7 | Global import-time warning suppression in four modules | scoped, counted handling; subprocess test |
| F8 | Phase 5 text said the registered workflow runs unauthenticated files with a label; it refuses them (exit 3) | corrected in `DATA_AUTHENTICATION.md`; Phase 5 text kept as history |
| F9 | README quoted "551 passing" | corrected to the measured 944 passed, 26 skipped (section 6) |
| F10 | Provenance sidecars lacked intended use, split identifiers, performance link | contract 1.1.0; shipped artifacts remain `INCOMPLETE_DECLARED` |
| F11 | No machine-readable experiment or claims record | registers + tests |
| F12 | No CI | workflow added; not executed remotely |
| Open | F1 tie-dropping count, 31 vs 32 subjects, `test_time` skew, label noise from interpolated UPDRS | need the data (D4-D6) |

## 6. Test results

| Run | Result |
|---|---|
| Historical (Phase 4A, `8e4f4d3`, clean worktree) | 764 passed, 26 skipped |
| Phase 5 full suite | 864 passed, 26 skipped, 79 warnings |
| Phase 6, first full run (before the registers were reconciled) | **1 failed**, 943 passed, 26 skipped |
| **Phase 6 full suite (final)** | **944 passed, 26 skipped, 83 warnings** |

* The one failure in the first run was `test_the_untraceable_0_6178_figure_is_nowhere_in_the_code_or_metadata` (Phase 5): my new `claims_register.json` quoted the untraceable
  figure while recording that it is unsupported. The test is correct (the figure must not appear in code or metadata); I reworded the claim to point to
  `docs/phase5/BASELINE_AUDIT.md` instead. The test was not changed.
* +80 passing tests versus Phase 5 (`tests/phase6/*` and the pinned planning-table test). No test was weakened, skipped or removed; the two Phase 5 provenance
  tests that needed a validation-run link were updated for contract 1.1.0 (see `ARCHITECTURE_AUDIT.md` section 2).
* The 26 skips are unchanged: 22 RLS tests (no separate Supabase test project) and 4 live Supabase/Gemini tests. No test depends on the real UCI files.
* Warnings 79 to 83. Same four types as Phase 5 (scikit-learn `InconsistentVersionWarning` on loading artifacts pickled under 1.9.0, a small-class `UserWarning`,
  a `fork()` multi-thread `DeprecationWarning` in a pre-existing test, and a hypothesis-plugin notice) plus the starlette/httpx deprecation notice. No warning is
  raised from `tests/phase6`. The per-instance increase was not triaged further. No warning was suppressed.
* The end-to-end report-pipeline test takes about 2.5 minutes and runs on forged data only.
* Registers: `python -m ml.validation.experiment_registry check` and `python -m ml.validation.claims_register check` both exit 0.

## 7. Remaining blockers

| # | Blocker | Owner |
|---|---|---|
| 1 | Authorised copies of UCI 174 and 189 (or a granted local folder holding them) | user |
| 2 | Pinning dataset hashes after fingerprint authentication | human with the files |
| 3 | Clinical decisions: intended use, endpoint, acceptable errors, thresholds (`REQUIRES_CLINICAL_INPUT`) | qualified clinicians |
| 4 | Independent external cohort and approvals | clinical / organisational |
| 5 | GitHub push (403; not authorised; never retried) and first remote CI run | user |

## 8. Next steps and exact commands

On a machine that can reach UCI (or with the files attached to this conversation):

```
cd backend
PYTHONPATH=. python -m ml.validation.dataset_manifest status
PYTHONPATH=. python -m ml.validation.workflow run --disease-csv /path/parkinsons.data --risk-csv /path/parkinsons_updrs.data --out ml/validation/results/workflow
PYTHONPATH=. python -m ml.validation.workflow run --candidate-set p6-1-ext --disease-csv /path/parkinsons.data --risk-csv /path/parkinsons_updrs.data --out ml/validation/results/workflow_p6ext
PYTHONPATH=. python -m ml.validation.phase6_report --p4a-dir ml/validation/results/workflow --ext-dir ml/validation/results/workflow_p6ext --risk-csv /path/parkinsons_updrs.data --out ml/validation/results/phase6/report
PYTHONPATH=. python -m ml.validation.experiment_registry check
PYTHONPATH=. python -m ml.validation.claims_register check
git push -u origin phase2/advanced-ml-hardening      # from a machine with GitHub access
```

Then: read the exclusion and subject reports, settle D4-D6, pin hashes, update the registers (a new real-data run must be added to
`EXPERIMENTS` and its claims reclassified by the checker, not by hand), and only then plan external validation with clinicians
(`CLINICAL_VALIDATION_PROTOCOL.md`).

## 9. Statement on claims

No document or test in Phase 6 asserts clinical validity, external or prospective validation, calibration, fairness or clinical utility.
`test_no_phase6_document_claims_clinical_validity_without_negation` fails if such a claim is written without negation.
