# Phase 6: External Validation Protocol (draft for clinical review)

**Status: `BLOCKED`.** No independent cohort exists in this repository or was accessible. This is a concrete data-acquisition and
analysis plan to be reviewed, completed and registered by qualified people **before any outcome data are opened**. It is not a result
and not clinical validation, and it sets no performance target (`REQUIRES_CLINICAL_INPUT`).

## 1. Why internal evidence cannot substitute

Both public datasets are single-source, small (about 42 patients; 8 healthy subjects) and repeated-measures. Subject-grouped nested
cross-validation estimates generalisation to *new subjects from the same source*. It cannot detect shift in device, site, language,
recording protocol, disease stage or population. Only an independent cohort can.

## 2. Candidate cohort requirements

| Requirement | Specification |
|---|---|
| Independence | different site(s), different recording devices/protocol where possible, no subject overlap with UCI 174/189 (checked by the data owner) |
| Disease model | sustained-vowel recordings, confirmed diagnosis by a movement-disorders clinician, age/sex-matched healthy controls, medication state recorded |
| Risk model | repeated clinic UPDRS assessments (real, not interpolated) paired with same-day voice recordings; baseline visit identified |
| Labels | assessed independently of the model; the endpoint definition fixed in advance by clinicians |
| Governance | ethics/IRB approval or documented exemption, data-use agreement, de-identification, retention plan |
| Eligible mirrors | none unless the owner provides provenance; unverified mirrors are not substituted |

## 3. Freeze before opening the cohort

The following are frozen, hashed and recorded in a new protocol file (`p7-1.json` or later; never an edit of `p4a-1.json` / `p6-1.json`):

1. Model artifact SHA-256 and training provenance record (`artifact_contract.py` 1.1.0), preprocessing, feature schema and units.
2. Decision threshold(s) and their rationale, chosen from **training-side** data and clinical input only.
3. Primary endpoint, primary metric and analysis plan, including handling of abstentions (count them; do not drop them).
4. Subgroups, minimum sizes, multiplicity handling, and the rule for declaring success or failure.

If any component is tuned on this cohort, the result is `ADAPTATION`, not `EXTERNAL VALIDATION`, and is labelled so. Recalibration
on the cohort is reported separately from the frozen-model result.

## 4. Analysis (frozen model, no refitting)

Primary: discrimination (AUROC) with subject-cluster bootstrap interval; sensitivity, specificity, PPV, NPV at the frozen threshold with
exact (Clopper-Pearson) or cluster-aware intervals; calibration intercept and slope; Brier; decision curve over a clinician-specified
threshold range. Secondary: subgroup performance above minimum size; comparison with the time/baseline-only reference models; abstention rate and reason.
All with the library in `ml/evalkit/clinical_metrics.py`.

## 5. Sample size (arithmetic, `PLANNING_ARITHMETIC_NOT_EVIDENCE`, `ml/validation/sample_size.py`)

Assumed sensitivity 0.85 and specificity 0.80 (placeholders, not expectations); Clopper-Pearson half-width:

| Prevalence | Half-width | Positives | Negatives | Total subjects |
|---|---|---|---|---|
| 10% | 0.10 | 49 | 62 | 490 |
| 10% | 0.05 | 196 | 246 | 1,960 |
| 25% | 0.10 | 49 | 62 | 196 |
| 25% | 0.05 | 196 | 246 | 784 |
| 50% | 0.10 | 49 | 62 | 124 |
| 50% | 0.05 | 196 | 246 | 492 |

AUC to within +/-0.05 (Hanley-McNeil): 631 subjects at AUC 0.70 and 25% prevalence (158 positive / 473 negative); 494 at AUC 0.80; 280 at AUC 0.90.
At 50% prevalence: 416, 302 and 158. With repeated recordings the effective sample is smaller: the design effect for 6 recordings per subject at ICC 0.5 is 3.5
(about 29 subjects' worth of information per 100 rows), and for about 139 per subject it is 70. Subject counts, not recording counts, are the planning unit.

## 6. Temporal and prospective stage

After retrospective external validation: a temporally separated or prospective evaluation with the same frozen artifact, a pre-registered analysis, and
monitoring of drift and abstention. Prospective deployment into care is a separate decision with its own risk assessment (`SAFETY_CASE.md`).

## 7. Dependencies and next step

Blocked on D8 and D9 (`DATA_AUTHENTICATION.md`): an independent cohort and clinical decisions. Next step for the project owner: identify a clinical
collaborator and a data custodian, share this document and `CLINICAL_INTENDED_USE.md`, and have them fill the `REQUIRES_CLINICAL_INPUT` fields
and approve the freeze list in section 3.
