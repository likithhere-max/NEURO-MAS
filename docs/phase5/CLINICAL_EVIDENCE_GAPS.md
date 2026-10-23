# Phase 5: Clinical Evidence Gaps

Each row is a gap between what exists and what a clinical claim would need. "Owner" says who can close it; most gaps are not
engineering work. No gap below has been closed by Phase 5; Phase 5 made several of them *measurable* and the rest *explicit*.

Legend: **OPEN** = nothing exists; **BLOCKED** = engineering is ready, external dependency missing; **PARTIAL** = some
engineering evidence, no clinical evidence.

| # | Gap | Status | What exists | What is needed | Owner |
|---|---|---|---|---|---|
| 1 | Reproduction of historical figures from the public data | **BLOCKED** | registered workflow, authentication, NOT_RUN manifest, 403 record | authorised download of both files (or user-supplied copies), then `workflow run` | data holder / user |
| 2 | Dataset hash lock | **BLOCKED** | `DATASET_MANIFEST.json` (unpinned), `pin` command requires authentication | files to hash | data holder |
| 3 | Resolution of 5,801 vs 5,833 and 31 vs 32 | **BLOCKED** | mechanism tests, exclusion and subject reports | the files | data holder |
| 4 | Size of the `test_time` train/serve skew | **BLOCKED** | single convention, tests, moments recorded | the files | data holder |
| 5 | Does voice add information beyond time and baseline UPDRS? | **BLOCKED** | frozen decision rule; power planning (about +0.08 AUC detectable, optimistic) | the files, then the registered run | engineering after data |
| 6 | Subject-level uncertainty for every headline metric | **BLOCKED** | cluster bootstrap and Clopper-Pearson in the protocol | the files | engineering after data |
| 7 | Independence of evidence: external cohort | **OPEN** | none | independent cohort, ethics approval, data-use agreement, locked model and protocol | clinical research team |
| 8 | Prospective / temporally separated evaluation | **OPEN** | none | prospective study design and approval | clinical research team |
| 9 | Clinical definition of "worsening" | **OPEN** | provisional 3.0-point threshold on interpolated UPDRS | clinician-defined, validated endpoint; real (not interpolated) assessments | clinicians |
| 10 | Forecasting (the risk label is concurrent) | **OPEN** | none; documented as a nowcast | longitudinal design with a defined horizon | clinicians + engineering |
| 11 | Target population, setting, recording conditions | **OPEN** | none | intended-use decision (`CLINICAL_INTENDED_USE.md`) | clinicians / organisation |
| 12 | Prevalence-appropriate performance (PPV/NPV) | **OPEN** | arithmetic only, not evidence | target-population prevalence and cohort | clinicians |
| 13 | Calibration | **OPEN** | class-weighted scores, not calibrated | calibration on independent data at target prevalence | engineering after data |
| 14 | Subgroup performance and fairness (age, sex, severity, device, language) | **OPEN** | none; infeasible with 8 healthy subjects | adequately sized, characterised cohorts | clinical research team |
| 15 | Healthy-control adequacy | **OPEN** | 8 healthy subjects; exact specificity interval for 8 units is very wide | larger, matched control group | clinical research team |
| 16 | Comparison with clinician performance / usual care | **OPEN** | none | reader study or pathway comparison | clinicians |
| 17 | Clinical utility (does using it improve outcomes?) | **OPEN** | none | utility study | clinicians |
| 18 | Safety analysis of failure modes in the care pathway | **PARTIAL** | software abstention and refusal behaviour tested | hazard analysis with clinicians (`SAFETY_AND_HUMAN_OVERSIGHT.md`) | clinicians + safety |
| 19 | Regulatory classification and quality system | **OPEN** | none | regulatory assessment | organisation |
| 20 | Shipped-artifact provenance | **PARTIAL** | sidecars declare fields unrecoverable (`INCOMPLETE_DECLARED`) | retrain on authenticated data via `train_final.py` to obtain `COMPLETE` records | engineering after data |
| 21 | Joint out-of-domain detector error rates | **BLOCKED** | detector and evaluation function exist | real data to measure false rejection and missed-OOD rates | engineering after data |

## Statements that must not be made on the current evidence

* "The model detects / predicts / screens for Parkinson's disease."
* "The risk model predicts worsening" (it is concurrent and unreproduced).
* "Voice features improve prediction" (not tested against time/baseline; the historical model was not compared with them).
* Any accuracy, AUC, sensitivity or specificity without its evidence class (`REAL_DATA_AUTHENTICATED...`, `UNAUTHENTICATED...`,
  `SYNTHETIC_FIXTURE_NOT_EVIDENCE`, or `NONE`) and uncertainty.
* "Externally validated" or "prospectively validated".
* "Clinically ready", on any engineering criterion.
