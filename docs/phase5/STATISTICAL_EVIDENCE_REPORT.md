# Phase 5: Statistical Evidence Report

**Evidence status of this document: there are no new empirical results from real data.** The real data files could not be
obtained (403), so the registered `p4a-1` experiments are `NOT_RUN`. This report contains (a) arithmetic on the historical,
unreproduced figures, and (b) design-stage planning calculations on assumed effects, labelled
`DESIGN_STAGE_PLANNING_NOT_EVIDENCE` in `backend/ml/validation/results/phase5/power_planning.json`. Neither is evidence that a
model works. Nothing here is clinical validation.

Reproduce: `cd backend && PYTHONPATH=. python -m ml.validation.power_planning --reps 20`.

## 1. What the historical figures can support

Independent units are subjects, not rows.

| | Disease | Risk |
|---|---|---|
| Rows | 195 | 5,801 |
| Independent units | 32 (metadata) or 31 (publisher) subjects; **8 healthy** | 42 patients |
| Winner accuracy minus majority accuracy | +0.0205 (4 recordings) | +0.0122 (71 rows) |
| Winner specificity (row-level) | 0.4375 (21 of 48) | 0.6503 |
| Winner sensitivity (row-level) | 0.8844 | 0.5982 |
| Pooled AUC | 0.7941 | 0.6405 |

Row-level Wilson intervals computed from the confusion matrices (disease specificity [0.307, 0.577]; risk specificity
[0.635, 0.666]) are **naive lower bounds on the true uncertainty**, because recordings of one subject are correlated and
the model was chosen on the same folds. For disease the honest unit for specificity is 8 healthy subjects: the exact interval
for k correct of 8 is wide (for example 4/8 gives [0.157, 0.843]; 8/8 gives [0.631, 1.0]). The artifact audit's table covers k = 0..8.

Reading: the disease winner beats always-predicting-PD by four recordings and misclassifies more than half of the healthy
recordings. The risk winner beats the majority rate by 1.2 percentage points. Neither difference can be called real or
meaningful without subject-level uncertainty, which only the registered run produces.

## 2. Precision available in principle (disease, analytic)

Hanley-McNeil with 24 PD and 8 healthy subjects, one score per subject, scores treated as fixed. Every figure is optimistic
(training variability, selection and repeated recordings ignored; normal approximation is poor with 8 negatives).

| Assumed true AUC | SE | 95% interval |
|---|---|---|
| 0.6 | 0.113 | [0.379, 0.821] |
| 0.7 | 0.100 | [0.503, 0.897] |
| 0.8 | 0.081 | [0.641, 0.959] |
| 0.9 | 0.055 | [0.793, 1.000] |

Null SE of the AUC is 0.120; the minimum detectable AUC against 0.5 at 80% power is about **0.835**. With this sample an AUC
below roughly 0.84 cannot be reliably told apart from chance, even optimistically. The historical 0.794 sits below that
threshold.

## 3. Precision available in principle (risk voice value, simulation)

Question: could the registered design detect that voice deltas add AUC beyond `test_time` + baseline UPDRS?

Setup: synthetic cohorts of 42 patients x 139 sessions with a *generated* voice signal of controlled strength; the
**lighter** configuration (5 outer / 3 inner folds, 200 bootstraps, LR with C in {0.1, 1}, one seed per repetition, 20
repetitions per level). It is not the registered 10-fold/5-seed procedure and ignores the two-reference, five-seed
intersection rule, so power under the actual rule would be **lower**.

| Generator voice effect | Mean AUC difference (FULL - TIME_BASELINE) | Interval excludes 0 on the positive side | Interval entirely below 0 |
|---|---|---|---|
| 0 (null) | -0.008 | 0 / 20 | 6 / 20 (30%) |
| 0.25 | +0.001 | 0 / 20 | 1 / 20 |
| 0.5 | +0.027 | 4 / 20 (20%) | 0 / 20 |
| 1.0 | +0.081 | 16 / 20 (80%) | 0 / 20 |
| 2.0 | +0.158 | 20 / 20 | 0 / 20 |

Reading: roughly **+0.08 AUC** was the smallest tested effect detected in 80% of repetitions (single-seed, single-reference).
Effects of +0.03 or less are mostly invisible. Under the null, adding noise features lowered AUC by about 0.008 on average
and the interval fell entirely below zero in 30% of repetitions, so a "negative" interval is not a sign that voice is harmful;
it can reflect added noise dimensions. The generator unit is arbitrary: only the AUC difference transfers, and real power may
be higher or lower because real labels derive from interpolated UPDRS, real voice is correlated within patient, and the real
noise structure is unknown. With 20 repetitions the detection rates carry about +/-22 points of Monte-Carlo error.

Consequence for interpretation: a registered result of "the interval includes 0" will mean *the data cannot show that voice
adds information*, not *voice adds nothing*.

## 4. Prevalence sensitivity (arithmetic only)

PPV/NPV computed from the historical sensitivity and specificity (unverified figures) for illustrative prevalences. These are
**not** estimates for any real population; the training prevalences (0.754, 0.382) are artefacts of the datasets.

| Prevalence | Disease PPV / NPV | Risk PPV / NPV |
|---|---|---|
| 1% | 0.016 / 0.997 | 0.017 / 0.994 |
| 5% | 0.076 / 0.986 | 0.083 / 0.969 |
| 10% | 0.149 / 0.971 | 0.160 / 0.936 |
| 25% | 0.344 / 0.919 | 0.363 / 0.829 |
| 50% | 0.611 / 0.791 | 0.631 / 0.618 |
| 75% | 0.825 / 0.558 | 0.837 / 0.350 |

At low prevalence PPV collapses; no deployment prevalence is known for either task, so no PPV is claimed.

## 5. Multiplicity and independence

* Registered claim rule: voice value requires the paired interval of FULL minus each of two references to exclude 0 for every
  seed (an intersection; conservative against false positives, costly in power).
* Anything beyond that rule (extra feature sets, subgroups, additional endpoints, threshold sweeps) is exploratory and is not
  adjusted for multiplicity.
* Five seeds are repeats of the split on one sample, not replications.

## 6. What the registered run will report (when authorized data exist)

Per model: subject-level and row-level metrics with cluster-bootstrap intervals, Clopper-Pearson intervals for
sensitivity/specificity, per-fold AUCs, baseline comparisons with paired differences, and a permutation test (disease). The
generated report (`report_from_artifacts.py`) can only print numbers that appear in the machine-readable results as
registered claims, and a tampered number is detected by test. Reports from unauthenticated files carry
`UNAUTHENTICATED_USER_FILE_NOT_A_REPRODUCTION_OF_THE_REPORTED_FIGURES`.

## 7. Bottom line

The statistics available today say: small independent sample (8 healthy subjects; 42 patients), gains over trivial baselines
of 0.02 and 0.01 in accuracy, a disease AUC below the minimum detectable level for the design, and a voice-adds-information
question that the design can only answer for effects around +0.08 AUC or larger. These are reasons for caution, not findings
about the models' real behaviour.
