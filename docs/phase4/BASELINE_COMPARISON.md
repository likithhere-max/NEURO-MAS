# NEURO-MAS Phase 4A: Baseline Comparison

**Read this first.** The only baseline numbers in this document are those that follow **arithmetically** from the shipped models'
own metadata (majority-class accuracy). **No time-only, baseline-only, demographic-only or classic-feature baseline has been computed
on real data**, because the data could not be obtained (`DATA_PROVENANCE.md`). They are listed as `NOT RUN`, not estimated.
Nothing below is a statement about clinical usefulness.

## 1. What can be said without the data

| | Disease classifier | Risk classifier |
|---|---|---|
| Rows / subjects | 195 / 32 (publisher: 31) | 5,801 / 42 |
| Prevalence of the positive class | 147/195 = 0.7538 | 2,215/5,801 = 0.3818 |
| **Majority-class accuracy** (arithmetic) | **0.7538** | **0.6182** |
| Shipped winner, pooled accuracy | 0.7744 | 0.6304 |
| Winner minus majority | **+0.0205** (4 recordings of 195) | **+0.0122** (71 rows of 5,801) |
| Shipped winner, pooled ROC-AUC | 0.7941 | 0.6405 |
| Confusion matrix (winner) | TP 130, FN 17, FP 27, TN 21 | TP 1,325, FN 890, FP 1,254, TN 2,332 |
| Sensitivity / specificity (recording or row level) | 0.884 / **0.4375** | 0.598 / 0.650 |
| Balanced accuracy | 0.661 | 0.624 |

Reading these honestly:

* The raw accuracy of both shipped models is barely above "always predict the majority class". The risk model's accuracy gain is 1.2 percentage points; the disease model's is 2.1.
* The disease winner labels 27 of 48 healthy recordings as PD (specificity 0.4375). It looks good on accuracy mostly because 75% of recordings are PD.
* A pooled ROC-AUC of 0.64 is weak discrimination. The reported figures carry **no confidence interval**, and (per the Phase 2 record) the winning candidate was chosen on the same folds that produced the reported score, which inflates it.
* These figures were not recomputed from data. They were checked for internal consistency (they reconstruct to integer confusion matrices) and nothing more.

## 2. Registered baselines (protocol `p4a-1`, SHA-256 `0b2f10a7…723e42`)

Every baseline is evaluated with the **same** subject-grouped nested protocol as the model, so differences are not an artifact of a different split.

### Risk task (outer 10-fold grouped by patient, 5 seeds, inner 5-fold grouped, selection by inner ROC-AUC)

| Feature set | Content | Why it is there | Real-data result |
|---|---|---|---|
| `FULL` | 16 voice deltas + `test_time` + `baseline_total_UPDRS` | the shipped model's inputs | **NOT RUN** |
| `VOICE_ONLY` | 16 voice deltas | does voice alone carry anything | **NOT RUN** |
| `TIME_BASELINE` | `test_time`, `baseline_total_UPDRS` | the label is a change score; elapsed time and starting severity may explain it | **NOT RUN** |
| `TIME_ONLY` | `test_time` | simplest progression proxy | **NOT RUN** |
| `BASELINE_ONLY` | `baseline_total_UPDRS` | regression-to-the-mean type effects | **NOT RUN** |
| `DEMOGRAPHIC_ONLY` | `age`, `sex` | demographic-only baseline (columns exist in the dataset) | **NOT RUN** |
| `TIME_DEMOGRAPHIC` | time + age + sex | | **NOT RUN** |
| `TIME_BASELINE_DEMOGRAPHIC` | time + baseline + age + sex | strongest voice-free reference | **NOT RUN** |

**Pre-registered rule for "voice adds information":** the 95% patient-cluster-bootstrap interval of AUC(FULL) − AUC(reference) must exclude 0
for **both** `TIME_BASELINE` and `TIME_BASELINE_DEMOGRAPHIC` and the effect must reproduce for **every** seed. Otherwise the correct statement is
"the data cannot show that voice adds anything beyond time and baseline", which is *not* the same as "voice adds nothing".

The shipped model's largest standardised coefficients are `test_time` (0.93) and the jitter deltas (−1.71 to +1.16, mixed signs, collinear), and its
`baseline_total_UPDRS` coefficient is −0.34. That is suggestive that elapsed time carries much of the signal, but coefficients of a
collinear model are not evidence of what drives its AUC; only the `FULL` vs `TIME_BASELINE` comparison can answer it, and that comparison is **not run**.

### Disease task (outer leave-one-subject-out, 5 seeds)

| Baseline | Content | Real-data result |
|---|---|---|
| `dummy_prior` | constant, no information | **NOT RUN** (its chance values are known by definition: AUC 0.5, PR-AUC = prevalence 0.7538, accuracy 0.7538) |
| `classic3_LR` | logistic regression on three classical perturbation measures | **NOT RUN** |
| Time-only / demographic-only | **Not applicable.** The disease file has no time or demographic columns; no such baseline was or can be built, and none is reported. |

The disease comparison reports both recording-level and **subject-level** metrics, and the subject-level specificity is based on 8 healthy
subjects. With k of 8 healthy subjects correct, the exact 95% interval is wide however the model does:

| k correct / 8 | 0 | 2 | 4 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Clopper–Pearson 95% | 0–0.369 | 0.032–0.651 | 0.157–0.843 | 0.349–0.968 | 0.473–0.997 | 0.631–1.000 |

Even a perfect 8/8 leaves the lower bound at 0.63.

## 3. A trap in the baselines, found and fixed in Phase 4A (defect E04b)

A constant-prior ("no-skill") model scored with leave-one-subject-out has a **pooled AUC of exactly 0, not 0.5**. Each held-out subject
is scored with the prevalence of the *other* subjects, and a held-out PD subject lowers that prevalence, so every PD subject scores below every
healthy subject. Quoting that 0.0 next to the classifier's AUC would make any real model look good for no reason. The runner now reports the
no-skill baseline's ranking metrics as a constant score's (AUC 0.5, PR-AUC = prevalence) and keeps the raw artifact value under `loso_prior_artifact`,
clearly labelled. Test: `tests/phase4/test_p4_protocol_integrity.py::test_dummy_prior_leave_one_subject_out_artifact_is_real_and_is_kept_but_labelled`.

The same mechanism pulls **every** pooled-LOSO AUC below its true value under the null (Phase 3, finding S4), so pooled LOSO AUCs are conservative
under no signal and are compared only against baselines scored the same way, plus a subject-label permutation test.

## 4. Second guard found in Phase 4A (defect E04a)

The exact-duplicate-row leakage guard treated two *different* patients with identical `(age, sex)` as a recording duplicated across the train/test
boundary, so the registered run would have **crashed at its first demographic baseline on real data**. The guard now runs on the multi-column voice
sets (where a duplicate is meaningful) and is off for the tiny baseline sets. Folds do not depend on the features, so the voice-set check covers
identical folds. Tests: `test_demographic_only_baseline_does_not_crash_…` and `test_duplicate_guard_still_fires_for_the_voice_feature_sets`.

## 5. Commands

```
cd backend
PYTHONPATH=. python -m ml.validation.phase4a artifacts                                   # the arithmetic in section 1; no data needed
PYTHONPATH=. python -m ml.validation.phase4a run --kind risk    --csv parkinsons_updrs.data   # all 8 risk feature sets, 5 seeds
PYTHONPATH=. python -m ml.validation.phase4a run --kind disease --csv parkinsons.data         # dummy_prior, classic3_LR, shipped candidates
```

Without the data files these commands stop with exit code 2 and write `…_NOT_RUN.json`; they never produce numbers.
