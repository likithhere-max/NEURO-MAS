# Phase 5: Validation Protocol Audit (`p4a-1`)

Question: is the frozen Phase 4A protocol sound enough to produce trustworthy evidence once authorized data arrive, and has it
been kept unchanged? Short answer: **no estimand-changing defect was found; protocol `p4a-1` is unchanged**; several
limitations are documented below so they are not mistaken for strengths.

## 1. Immutability

* The protocol is frozen in `backend/ml/validation/protocol_registry/p4a-1.json` (status `FROZEN`, frozen 2026-10-23, before
  any real result existed).
* SHA-256 `0b2f10a7ec885419d9a2e72ad80b9d4c60b76987ea71e654edcb157b86723e42` is pinned as a literal in a test
  (`test_protocol_hash_is_pinned_to_the_value_registered_before_any_result`), so editing the protocol and its registry copy
  together still fails.
* `test_protocol_equals_its_frozen_registry_copy` and `test_every_registry_file_hash_matches_its_content` cover drift.
* Amendment rule recorded in the registry: *never edit the file; a change is a new file (`p4a-2.json`) stating the defect, the
  rationale, the changed estimand or procedure, and whether the resulting analysis is exploratory or confirmatory.*
* Every workflow run writes the protocol id and hash into `experiment_manifest.json`.

Changes to protocol **code** since freeze, none of which change the hash or the estimand: E04a (duplicate guard off for tiny
demographic baselines) and E04b (no-skill ranking metrics for a constant score), both found by pre-data tests in Phase 4A, and an
observe-only `recorder` hook in `phase4a.py` that stores predictions and fold assignments without influencing any fit.

## 2. What the protocol specifies

| Element | Disease (PD vs healthy) | Risk (concurrent worsening) |
|---|---|---|
| Independent unit | subject, parsed from the recording name | patient (`subject#`) |
| Outer evaluation | leave-one-subject-out | 10-fold subject-grouped, repeated over 5 seeds (base 20261023) |
| Selection | nested; inner grouped CV on training subjects only | inner 5-fold grouped, training patients only |
| Preprocessing | scaler and selector fitted inside training folds | scaler fitted inside training folds |
| Baselines | constant prior; 3-feature LR | 8 registered feature sets incl. TIME_ONLY, BASELINE_ONLY, TIME_BASELINE, demographic sets |
| Threshold | 0.5 (fixed, no tuning on evaluation data) | 0.5 |
| Uncertainty | cluster (subject) bootstrap, Clopper-Pearson, permutation test | cluster bootstrap, paired AUC difference intervals |
| Voice-value rule | n/a | paired 95% interval of FULL minus **each** reference must exclude 0 for **every** reference and **every** seed |

## 3. Leakage checks (tested)

* No identifier, label or outcome column appears in any registered risk feature set (`test_registered_risk_feature_sets_contain_no_identifier_label_or_outcome_column`).
* A visit's features depend only on that visit and the patient's first visit (`test_risk_features_of_a_visit_depend_only_on_that_visit_and_the_patients_first_visit`); the label uses only the current and first visits.
* Each subject appears in exactly one test fold per model and seed, never in train and test together
  (`test_fold_assignments_cover_every_subject_exactly_once_per_model_and_seed`).
* Phase 4A deliberately constructed leaky pipelines and showed the protocol separates them from the honest one on synthetic
  data labelled `SYNTHETIC_FIXTURE_NOT_EVIDENCE` (trap AUC 0.9997 versus the protocol's 0.459; the permutation test rejected
  in only 1 of 24 runs on that scenario - recorded in Phase 4A as a limit of the test, not a success).
* Pipeline stress tests (`test_p5_pipeline_stress.py`): missing values, non-finite values, unexpected label categories,
  duplicated records, unparseable names, a subject with two labels, changed schema/units, row-order and float-format changes,
  out-of-order and duplicate risk records, a temporal shift in `test_time`, and a changed class prevalence. Each is either
  refused with a reason, handled deterministically and reported, or visible in the report. None is silently repaired.

## 4. Limitations of the protocol (not defects, not hidden)

1. **Risk has no registered prevalence-only baseline.** Majority accuracy is analytic (0.6182 from the shipped class counts);
   the data-based figure will come from the TIME/BASELINE sets. Adding a prevalence model would need an amendment.
2. **Pooled out-of-fold AUC mixes fold-specific score scales** (and, for LOSO, 32 different models). Per-fold AUCs are
   reported alongside; pooled LOSO AUC is biased below 0.5 under the null for a constant model (E04b).
3. **The permutation test covers a fixed pipeline** and not the full selection procedure, so it is not a test of "the whole
   modelling process has signal".
4. **Seeds are not independent datasets.** The five seeds vary fold assignment on one sample; agreement across seeds
   measures split sensitivity, not replication. Do not multiply them into a sample size.
5. **Class-weighted scores are not calibrated probabilities.** No calibration claim is made or tested; no calibration was
   fitted on evaluation data.
6. **Effective sample size** is about 42 patients (risk) and 8 healthy subjects (disease). Row counts (5,801 / 195) are not
   independent observations.
7. **The risk label is concurrent and derived from interpolated UPDRS.** It tests association within an observation window, not
   forecasting; a smooth function of time makes `test_time` a strong competitor to voice by construction.
8. **The intersection voice-value rule is conservative** (two references x five seeds must all exclude 0). It protects against
   a false "voice helps" claim at the cost of power (see `STATISTICAL_EVIDENCE_REPORT.md`).
9. **Multiplicity**: comparisons beyond the registered rule, subgroup analyses and any added endpoint are exploratory and are
   not adjusted.

## 5. Evidence separation

Four evidence classes must stay separate in every report, and the workflow enforces labels rather than trusting authors:

| Class | Status in this repository |
|---|---|
| Exploratory (the historical metrics) | exists, not reproduced |
| Confirmatory (registered `p4a-1` on authenticated data) | **not run** (no data) |
| External validation (independent cohort) | **does not exist** |
| Prospective | **does not exist** |

A run on a file that cannot be authenticated against the shipped artifact is labelled
`UNAUTHENTICATED_USER_FILE_NOT_A_REPRODUCTION_OF_THE_REPORTED_FIGURES`. Internal cross-validation is never called external.

## 6. Verdict

The protocol is adequate for what it can show (internal, subject-grouped, uncertainty-aware comparison with fair baselines
on public data) and inadequate for anything beyond that. It cannot establish generalisation to new sites, devices, recording
conditions or populations; it cannot establish clinical benefit. Any defect discovered after real data are seen requires a
versioned amendment before confirmatory use; the first confirmatory run is the only one allowed to count as such.
