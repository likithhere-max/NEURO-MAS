# NEURO-MAS Phase 4A: Rigorous ML Validation

## 0. Verdict (read this first)

**The question was: do the existing models give reproducible predictive value beyond simple baselines? Phase 4A cannot answer it, because the
training data could not be obtained.** Both UCI downloads fail with `403 Forbidden`; no legitimate mirror was reachable; no hash lock exists.
Consequently:

* **No model figure was re-computed.** The figures in the Phase 4A brief (42 subjects, AUC 0.640, accuracy 0.630 vs majority 0.618) are those of the **risk** classifier. They are internally consistent with the shipped metadata but remain **unverified against data**.
* **No time-only, baseline-only or demographic-only baseline has been computed on real data**, so nothing is claimed about whether voice features add anything. Only arithmetic majority-class baselines are reported.
* **Neither model is shown to be clinically useful, and nothing here supports clinical use.** On the evidence that exists (the models' own metadata), both beat "always predict the majority class" by about 1–2 accuracy points, with no confidence intervals and with model selection on the scoring folds.
* What Phase 4A *did* deliver: a **pre-registered, leakage-resistant evaluation protocol** that fails clearly without data, tested on adversarial synthetic data (labelled not-evidence); a way to **authenticate** a supplied dataset against the shipped artifacts; fixes for **four defects** (E01–E04); and the exact commands that produce every metric once the two files are supplied.

| Evidence | Status |
|---|---|
| Recomputation of reported figures from the original data | **Not possible here** (data inaccessible) |
| Artifact-derived checks (hashes, row counts, integer confusion matrices, majority baselines) | Done, section 2 |
| Real-data baselines (time-only, demographic-only, classic features) | **Not run** |
| Subject-grouped nested evaluation with CIs, seeds, permutation test, on real data | **Not run** (code complete and tested) |
| Evaluation machinery on synthetic data with known truth | Done, section 6, **NOT evidence about any model** |
| Prospective or clinical validation | None exists |

## 1. Inventory (Step 1)

No model is treated as validated because its metadata or an earlier report says so.

| | Disease classifier | Risk classifier |
|---|---|---|
| Artifact | `ml/models/parkinsons_demo_classifier.joblib` | `ml/models/parkinsons_risk_classifier.joblib` |
| Endpoint | `app/api/routes/ml_demo.py` (disease route) | same module (risk route) |
| Data | UCI Parkinsons (174) | UCI Parkinsons Telemonitoring (189) |
| Features | 22 acoustic measures; shipped model uses top 10 chosen by `SelectKBest` | 16 voice deltas (value minus that patient's first-visit value) + `test_time` + `baseline_total_UPDRS` |
| Label | PD vs healthy, per subject | `total_UPDRS(t) − first visit ≥ 3.0`, concurrent, row-level (provisional threshold, not a validated MCID) |
| Shipped pipeline | StandardScaler → SelectKBest(10) → RandomForest (200 trees, balanced); final fit on all 195 rows | StandardScaler → LogisticRegression (C = 0.5, balanced); final fit on 5,801 rows |
| Split used for the shipped figures | nested leave-one-subject-out (Phase 2 protocol) | grouped by patient (Phase 2 protocol) |
| Selection | winner chosen among 5 (disease) / 2 (risk) candidates **on the same folds that produced the reported score** | same |
| Records dataset hash / seed / training library version in metadata | no / no / only inferred (`MANIFEST.json`: 1.9.0) | no / no / only inferred |

Training code: `ml/training/train_parkinsons_classifier.py`, `ml/training/train_risk_classifier.py`. Evaluation code: `ml/evalkit/{splits,nested,metrics}.py`,
`ml/validation/{data,candidates,validate_models}.py`. All preprocessing sits inside sklearn `Pipeline`s; a structural test asserts that factories return fresh un-fitted pipelines.

## 2. Audit of the reported figures (Step 2)

### 2.1 Which model the brief's numbers belong to

"42 subjects, pooled AUC 0.640, accuracy 0.630, majority 0.618" are the **risk** model. The shipped risk metadata reconstructs to the integer
confusion matrix TP 1,325 / FN 890 / FP 1,254 / TN 2,332: accuracy 3,657/5,801 = **0.6304**, majority 3,586/5,801 = **0.6182**, AUC 0.6405. So
the brief's arithmetic is internally consistent. It could **not** be reproduced from data.

The disease model's figures are different: accuracy 0.7744 vs majority 0.7538, AUC 0.7941, specificity 0.4375 (confusion TP 130 / FN 17 / FP 27 / TN 21).

### 2.2 Checks that were possible, with results

| Check | Result |
|---|---|
| Artifact SHA-256 equals `MANIFEST.json` | yes (both) |
| Scaler `n_samples_seen_` equals metadata row count | 195 and 5,801, yes |
| Metadata reconstructs to integer confusion matrices | yes, all 7 candidate entries |
| Gain over majority class | +0.0205 (disease), +0.0122 (risk) |
| Risk rows: 5,875 − 42 first visits = 5,833 expected | **5,801 used: 32 unexplained** |
| Disease subjects | **32 in metadata vs 31 per publisher** |
| Confidence intervals in the shipped metadata | **none** |
| Model selection and scoring on the same folds | **yes** (optimistic selection bias) |

### 2.3 Can repeated measurements from one subject leak across folds?

* **In the shipped evaluation design:** the Phase 2 splitter groups by subject/patient and has guards that raise `LeakageError` on any shared group, shared row index or exact duplicate row across an outer boundary. These are covered by existing tests (`tests/test_ml_evalkit.py`, fold disjointness and fit-call spies) and by the Phase 4A tests below. So within the code path, leakage across subjects is guarded. **Whether the figures the metadata reports were produced by that code path is not verifiable** (no seed, version or data hash on record).
* **The leakage hazard is real and large.** Section 6 shows that on synthetic data with no label signal, a *row-level* split reaches AUC > 0.95 purely by recognising the subject's voice, while the subject-grouped protocol does not. Any prior figure built from a row-level split would be meaningless.
* **Residual hazard that grouping does not remove (risk task):** labels are derived from *interpolated* UPDRS, so within a patient they are a smooth function of time, and the effective number of independent assessments is far below 5,801. Grouping prevents leakage; it does not make 5,801 rows into 5,801 independent observations. Bootstrap intervals therefore resample **patients**, not rows.

### 2.4 Protocol that was registered before any real result exists

Frozen in `ml/validation/phase4a.py::PROTOCOL` (id `p4a-1`, SHA-256 `0b2f10a7ec885419d9a2e72ad80b9d4c60b76987ea71e654edcb157b86723e42`, written into every output):

* Outer split grouped by subject/patient: leave-one-subject-out (disease), 10-fold `StratifiedGroupKFold`-style grouped (risk). Inner selection 5-fold grouped, on the training rows of each outer fold only, by ROC-AUC.
* Scaling, feature selection, hyperparameter choice and model choice happen **inside** the training fold (pipelines). Threshold fixed at **0.5 in advance**; no threshold is tuned on evaluation data.
* 5 seeds (base 20261023) vary the fold assignment; dispersion across seeds and across outer folds is reported.
* Uncertainty: patient/subject **cluster bootstrap** (2,000) for AUC, PR-AUC, sensitivity, specificity; **exact Clopper–Pearson** for subject-level proportions; paired cluster bootstrap for AUC differences; subject-label **permutation test** (200) for the disease model.
* Calibration: reported only as a diagnostic; scores from class-weighted models are **not probabilities**. Brier skill is reported against constant prevalence.
* Interpretation rule fixed in advance (BASELINE_COMPARISON.md, section 2). The protocol was not altered after any result was seen; there are no real results.

## 3. Baselines (Step 3)

Documented in full in `BASELINE_COMPARISON.md`. Summary: majority-class baselines are known arithmetically (disease 0.7538, risk 0.6182) and the shipped
models exceed them by 0.0205 and 0.0122 in accuracy. The registered time-only, baseline-only, demographic-only and (disease) classic-feature baselines are
implemented, share the model's evaluation protocol, and are **NOT RUN on real data**. No baseline number other than the arithmetic ones is reported, and none is estimated.
The disease file has no time or demographic columns, so no such baseline exists for it. Class balance, subject and observation counts and label definitions are in `DATA_PROVENANCE.md`;
missingness is stated by the publisher as none and is **not verified**.

## 4. What the evaluation reports, and what it deliberately does not (Step 4)

When the data files are supplied the runner writes, per model and per seed, machine-readable JSON containing:

| Quantity | How | Notes |
|---|---|---|
| ROC-AUC, PR-AUC (PR-AUC chance level = prevalence) | pooled out-of-fold, plus per-outer-fold values | patient/subject **cluster bootstrap** 95% interval |
| Sensitivity, specificity, confusion matrix, prevalence | at the **pre-registered 0.5** threshold | cluster bootstrap intervals |
| Subject-level vs recording-level (disease) | subject score = mean of its recordings | subject-level counts with exact Clopper–Pearson; specificity rests on **8 healthy subjects** |
| Within-patient AUC (risk) | AUC computed inside each patient, then summarised | separates "ranks sessions of one patient" from "ranks patients" |
| Paired AUC difference vs each baseline | paired cluster bootstrap | the basis of the voice-value rule |
| Variation across folds and seeds | per-fold AUC list; AUC by seed with mean/SD/min/max | seeds change fold assignment and inner splits |
| Permutation test (disease) | subject labels permuted at subject level; a **fixed** pipeline (the modal nested choice) is refit each time | tests association only; it does **not** test the selection procedure. Reported with the null mean and 95th percentile; pooled LOSO AUC under the null sits **below** 0.5 |
| Calibration | reliability table + Brier skill vs constant prevalence | **diagnostic only**: scores come from class-weighted or tree models and are not probabilities. No calibration claim is made |
| PPV/NPV at stated prevalences | from sensitivity/specificity | illustrates that the disease model's PPV/NPV depend entirely on the assumed prevalence |

Small-sample limits that apply to every number the runner will produce: 42 patients (risk) and 8 healthy subjects (disease) are the independent
units; intervals are wide and are **not** adjusted for the model-selection step (the nested design makes the *point estimate* honest, not narrow);
a bootstrap with few clusters is itself unstable; multiple feature sets and seeds are compared, so an interval excluding zero on one comparison is not
a corrected significance claim. The report therefore states "excludes 0" or "does not exclude 0" and never "significant".

## 5. Defects found and fixed in Phase 4A

| ID | Defect | Fix | Test |
|---|---|---|---|
| E01 | The risk endpoint's disclaimer called the AUC "meaningfully better than chance" while the figure has no CI, was selected on its own folds, and was never compared with time/baseline-only models | wording replaced with those facts | `test_p4_inference_claims.py` |
| E02 | The API schema described `test_time` as days since the baseline visit; in the training data it is days since **recruitment** | schema text corrected; the size of the train/serve mismatch remains unquantified | `test_p4_inference_claims.py` |
| E03 | The Phase 2 runner labelled **any** csv of the right shape as `REAL_DATA_RUN` | dataset authentication against the artifact's scaler fingerprint; unauthenticated files get an explicit non-reproduction label | `test_p4_dataset_authentication.py` (10 tests) |
| E04a | The exact-duplicate guard aborted the registered run at the first demographic-only baseline (distinct patients share `(age, sex)`) | guard kept for voice feature sets, off for tiny baseline sets (folds do not depend on features) | two tests in `test_p4_protocol_integrity.py` |
| E04b | A constant-prior baseline scored leave-one-subject-out has pooled AUC **0**, not 0.5 | no-skill ranking metrics reported as a constant score's; the raw value is kept and labelled | `test_dummy_prior_leave_one_subject_out_…` |
| T1 | A Phase 3 property test (`test_auc_symmetries`) failed on a new hypothesis counterexample: `1 − p` rounds two distinct scores to a tie | precondition that the transform keeps scores distinct; the metric code was **not** changed | `tests/phase3/test_p3_properties.py` |

E04a was found by the Phase 4A integrity test written before the real data existed; with the data it would have crashed the registered run. E04b would have silently
placed an AUC of 0 beside the model's 0.79 in a baseline table.

## 6. Does the evaluation machinery itself behave? (synthetic; **NOT evidence about any model**)

Every number in this section comes from **randomly generated data with known ground truth** (`SYNTHETIC_FIXTURE_NOT_EVIDENCE`). It shows whether the *procedure*
resists the traps it exists to defeat. It says nothing about the Parkinson's models or data. 24 repetitions per scenario (seed base 4000, 40 permutations, 200 bootstraps);
with 24 repetitions a rate carries a margin of roughly ±20 percentage points, so rates below are indicative only. Raw per-repetition values are in
`backend/ml/validation/results/phase4a/selftest_SYNTHETIC_NOT_EVIDENCE.json`.

| Scenario | Truth by construction | Result | Reading |
|---|---|---|---|
| **A** Disease-like, strong per-subject voice fingerprints, labels random per subject | no usable signal | naive **row-level** K-fold mean AUC **0.9997**; the protocol (subject-grouped nested LOSO) mean subject-level AUC **0.459**; permutation test rejects at 0.05 in **1 of 24 (4.2%)** | the leakage trap is real and the protocol is immune. The protocol's value below 0.5 is the pooled-LOSO null bias already documented in Phase 3 |
| **B** Planted weak signal (shift 1.5 on 3 of 22 features, fingerprint SD 2.0) | weak signal | mean AUC 0.526; rejects in **2 of 24 (8.3%)** | **the protocol is nearly blind to a weak signal at 20 subjects.** Reported as run |
| **B2** Planted strong signal (shift 4.0), added after B showed low power | strong signal | mean AUC 0.922; rejects in **20 of 24 (83%)** | the procedure can detect a strong effect; B2 supplements B and does not replace it |
| **C** Risk-like, labels from time and baseline only, voice is pure noise | voice adds nothing | AUC(FULL) − AUC(TIME_BASELINE) mean **−0.028**; interval excludes 0 on the **positive** side in **0 of 24**; on the **negative** side in **9 of 24 (37.5%)**; contains 0 in 15 of 24 | never a false "voice helps"; adding noise columns does measurably hurt, so the interval is not a calibrated test of "difference = 0" and the registered rule is correctly one-sided |
| **D** Same, but voice carries real signal | voice helps | mean difference **+0.190**; interval excludes 0 on the positive side in **24 of 24** | the rule detects a real contribution |

What this supports: the evaluation does not reward subject recognition, does not produce false "voice adds value" conclusions when voice is noise, and detects a strong effect.
What it does **not** support: any statement about real power. Scenario B shows that with few independent subjects a modest true effect can go undetected, which is the
situation the real disease data (8 healthy subjects) is in. A "no significant difference" outcome on real data must therefore be read as inconclusive, not as evidence of no effect.
Wall time with shared CPU: 30 min 36 s.

## 7. Limitations

1. **No real-data result exists.** Everything the title promises (reproducibility of predictive value) is conditional on the two files being supplied.
2. **Effective sample sizes are small.** Disease: 8 healthy subjects; subject-level specificity cannot be known to better than a very wide exact interval. Risk: 42 patients, with labels built from interpolated clinic scores, so rows are highly dependent.
3. **The risk task is concurrent, not predictive.** The label is a change score at the same session as the features; the model is not shown to forecast anything.
4. **The 3-point threshold is provisional**, not a validated minimal clinically important difference.
5. **Possible dominance of non-voice features.** The shipped risk model's largest coefficient is elapsed time. Whether voice contributes anything beyond time and baseline is exactly the untested comparison.
6. **Train/serve skew** for `test_time` (recruitment vs baseline) is unquantified (E02).
7. **Calibration is not claimed.** Class-weighted logistic and tree scores are not probabilities.
8. **Multiple comparisons.** Eight feature sets, several candidates and five seeds are compared; intervals are not multiplicity-adjusted, and the pre-registered rule is the only decision statement made.
9. **Public research data only.** Healthy volunteers and a small telemonitoring cohort; no external, prospective or multi-site validation; no demographic subgroup analysis is possible.
10. **Artifact provenance gaps** (no dataset hash, seed, or recorded training library version; 32 unexplained rows; 31 vs 32 subjects).

## 8. Unresolved issues (in priority order)

1. Supply the two UCI files from a machine with access, record their SHA-256, run `authenticate` (a mismatch means the shipped figures are not reproduced by that file), then `run`.
2. Whether voice adds information beyond time and baseline (`FULL` vs `TIME_BASELINE`, `TIME_BASELINE_DEMOGRAPHIC`).
3. Whether the shipped disease model's specificity of 0.4375 and the risk AUC of 0.64 survive selection inside the training folds.
4. The 32 unexplained risk rows and the 31/32 subject discrepancy.
5. Quantify the `test_time` mismatch, or change the input definition to match training.
6. Retrain under the pinned library version and record dataset hash, seed and library version in the metadata.

## 9. Reproduce every reported statement

All commands from `backend/` with the project venv (Python 3.13.17; numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0).

| Statement | Command | Needs data |
|---|---|---|
| Hash/row-count/confusion-matrix/majority-baseline checks (sections 2, 3; BASELINE_COMPARISON §1) | `PYTHONPATH=. python -m ml.validation.phase4a artifacts` | no |
| Fail-clearly behaviour (exit code 2, `*_NOT_RUN.json`) | `PYTHONPATH=. python -m ml.validation.phase4a run --kind risk` (and `--kind disease`) with no file | no |
| Harness self-test (section 6) | `PYTHONPATH=. python -m ml.validation.phase4a selftest --out <dir>` (about 30 min) | no (synthetic) |
| Protocol hash | `PYTHONPATH=. python -c "from ml.validation import phase4a as P; print(P.PROTOCOL_SHA256)"` | no |
| Dataset authentication | `… phase4a authenticate --kind {disease,risk} --csv <file>` | yes |
| Disease validation incl. dummy/classic3 baselines, subject-level CIs, permutation test | `… phase4a run --kind disease --csv parkinsons.data` | yes |
| Risk validation incl. all 8 feature sets and voice-value rule | `… phase4a run --kind risk --csv parkinsons_updrs.data` | yes |
| Tests | `python -m pytest -q -p no:cacheprovider` (add `tests/phase4`) | no |

Recorded runs of this phase: `1 failed, 763 passed, 26 skipped` (9 min 36 s, CPU shared with the self-test) before fix T1; the final run is in section 10.

## 10. Final regression run

```
cd backend && python -m pytest -q -p no:cacheprovider
764 passed, 26 skipped, 67 warnings in 164.12s (0:02:44)      # 0 failed
```

Phase 3 ended at 714 passed / 26 skipped; Phase 4A adds 50 tests (`tests/phase4/`). The 26 skips are unchanged from Phase 3. The frontend was not touched in Phase 4A and its checks were not re-run.
The warnings are scikit-learn `InconsistentVersionWarning` (artifacts trained under 1.9.0, loaded under 1.9.1) and small-class fold warnings from synthetic fixtures.

**Nothing in this report claims or supports clinical validity, clinical usefulness or readiness for patients.**


---

**Later note (Phase 5, 2026-10-23).** This report is preserved unchanged as historical evidence. Phase 5 re-audited it in `docs/phase5/` (start at `FINAL_READINESS_REPORT.md`). Items resolved since: the single `test_time` convention is enforced (E02), provenance sidecars exist for the shipped artifacts, the registered workflow and dataset manifest exist. Items still open because the data cannot be downloaded here: reproduction of every figure above, the 5,801 vs 5,833 and 31 vs 32 discrepancies, and all real-data results (`NOT_RUN`). The 764 passed / 26 skipped result quoted historically is unchanged; the Phase 5 regression is recorded separately.
