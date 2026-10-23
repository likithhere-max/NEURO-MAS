# ML Experiment Protocol (Phase 2, B)

Pre-specified procedure for validating the two shipped demonstration models. It is written **before** any real-data
result exists, and the runner (`backend/ml/validation/validate_models.py`) implements exactly this. Nothing below was
tuned after seeing results (there are none).

Both models are **research demonstrations**. Passing every gate here would still not make either a clinical tool.

## 1. Questions
| Model | Task | Honest question |
|---|---|---|
| `parkinsons_demo_classifier` | PD vs healthy from 22 sustained-vowel acoustic features (UCI Parkinsons, id 174) | Does it separate **unseen people**, beyond a trivial rule, once model selection is nested? |
| `parkinsons_risk_classifier` | "worsening" = ΔtotalUPDRS ≥ 3 vs the patient's first visit (UCI Parkinsons Telemonitoring, id 189) | Do **voice deltas** add anything over time-since-baseline and baseline severity? (Concurrent state, not a forecast.) |

## 2. Unit of analysis, leakage control
* The unit is the **person**. Splits are by subject (disease: subject parsed from `phon_R01_S##_#`; risk: `subject#`).
* `ml/evalkit/splits.py` guards every split: group-disjoint, no shared row index, no exact-duplicate feature rows across the
  boundary, temporal order where applicable, group-label consistency. A violation raises `LeakageError` (the run is
  `failed`, not "mostly fine").
* Scaling, feature selection and every data-dependent step live **inside** the sklearn `Pipeline`, so they are re-fitted on
  each training split only (tested: `test_feature_selection_outside_the_fold_inflates_auc_on_pure_noise_but_pipeline_selection_does_not`,
  `test_scaler_inside_pipeline_is_fitted_on_training_rows_only`).
* Hyper-parameter "tuning" is expressed as separate named candidates, chosen by the **inner** loop on training rows only.

## 3. Nested cross-validation
| | Disease | Risk |
|---|---|---|
| Outer | leave-one-subject-out | GroupKFold(10) by patient |
| Inner (selection) | StratifiedGroupKFold(5) | StratifiedGroupKFold(5) |
| Candidates | LR (C = 0.1, 1), RF depth 3, RF full, RF + in-fold top-10 selection, small HistGB | LR (C = 0.01, 0.1, 1), small HistGB |
| Decision threshold | 0.5, fixed a priori (not tuned on test data) | 0.5, fixed a priori |
| Seed | 20261023 (override with `--seed`) | same |

If an inner loop cannot select (too few groups/classes), the fold records `selection_status != "selected"` and the count of such
fallbacks is reported.

## 4. Metrics (all in `ml/evalkit/metrics.py`)
Reported at **recording level and subject level** (disease). Undefined quantities are `null`, never 0 (single-class AUC,
empty denominators). ROC-AUC with a **cluster bootstrap CI over subjects** (default 1000 resamples; degenerate resamples are
counted and an interval is withheld if too many are unusable). Sensitivity/specificity with Wilson intervals. Brier score,
ECE and a reliability table. **PPV/NPV re-computed at prevalence 0.5 %, 1 %, 5 %, 20 %, 75 %** because the study prevalence (~75 % PD
in the disease file) is not a screening population.

## 5. Baselines and ablations (same nested protocol)
* Disease: no-skill (`dummy_prior`), a 3-feature classical model (`classic3_LR`: jitter %, shimmer, HNR), and a paired
  subject-bootstrap AUC difference nested-pipeline − `classic3_LR`.
* Risk: FULL vs voice-deltas-only vs `test_time + baseline_UPDRS` (no voice) vs `test_time` only vs `baseline_UPDRS` only vs
  dummy. The key quantity is `AUC(FULL) − AUC(test_time + baseline)` with a patient-bootstrap CI: if the interval contains 0
  the data cannot show that voice adds information (Phase 1 finding ML7 suspects regression-to-the-mean can explain the label).
* The **biased legacy protocol** (pick the best of the candidates on the same LOSO folds that score it) is computed and reported
  only to quantify its optimism.

## 6. Significance
Group-label permutation test of a fixed pipeline (the modal nested choice), permuting labels **between subjects** (not
rows). It tests association, not the selection procedure.

## 7. Pre-specified acceptance gates before the UI may show a probability from either model
1. Nested subject-level AUC: lower 95 % cluster-bootstrap bound > 0.5 **and** permutation p < 0.05.
2. Beats the no-skill baseline on Brier score.
3. Risk model: `AUC(FULL) − AUC(no-voice)` interval excludes 0; otherwise the voice claim is unsupported.
4. Calibration: if ECE > 0.1, show an ordinal/rank output, not a probability.
5. Artifacts load under the pinned scikit-learn version without version warnings (see §9).
If any gate fails, or the run did not happen, the endpoints stay labelled "software demonstration — not a screening tool"
(they are, and the OOD gate abstains on foreign inputs).

## 8. Reporting rules
Every JSON result contains `status` (`completed | failed | not_run`), `evidence_status`, `clinically_validated: false`, the
seed and the environment. `evidence_status = SYNTHETIC_FIXTURE_NOT_EVIDENCE` is written for code-path smoke tests and must
never be quoted. A failed or unreachable step writes `*_FAILED.json` and the report says "not run".

## 9. Artifact integrity (independent of data)
SHA-256 of each `.joblib` is checked against `ml/models/MANIFEST.json` before loading; loading then goes through a restricted
unpickler (`app/ml/safe_load.py`) that resolves only numpy / scikit-learn / `scipy.sparse` / joblib-array types. Neither is a
proof of safety (pickle remains pickle); the durable fix is a non-pickle format (ONNX / skops with trusted types) after re-training.

## 10. How to run (needs internet for step 1)
```bash
cd backend && pip install -r requirements.lock.txt
PYTHONPATH=. python -m ml.validation.validate_models fetch              # downloads UCI id 174 and 189, records SHA-256 (trust on first use)
PYTHONPATH=. python -m ml.validation.validate_models disease --n-boot 1000 --n-perm 100
PYTHONPATH=. python -m ml.validation.validate_models risk    --n-boot 1000 --n-perm 100
PYTHONPATH=. python -m ml.validation.validate_models artifacts
```
Results land in `backend/ml/validation/results/`. Then fill ML_VALIDATION_RESULTS.md §3 from those files verbatim.
