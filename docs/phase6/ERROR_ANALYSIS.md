# Phase 6: Error Analysis

**Status: `NOT_RUN`.** Error analysis needs held-out predictions, which need the data (`DATA_AUTHENTICATION.md`, D1).
This document specifies what will be produced and which failure modes are already known from the code and data description.
No error rate is stated because none was measured. No clinical validity is claimed.

## 1. Procedure (implemented in `ml/validation/phase6_report.py`, tested on forged data only)

For each model and seed, from the saved out-of-fold predictions:

1. Confusion matrix at the fixed 0.5 threshold with denominators, then sensitivity, specificity, PPV, NPV.
2. Calibration intercept and slope (from the metric library); recalibration is exploratory only.
3. **Per-subject error profile**: risk reports per-subject accuracy (minimum, median, maximum), the median per-subject AUC and the number of
   subjects whose own AUC is below 0.5; disease reports subject counts and the number of misclassified subjects. Subject-level rather than
   row-level counts, because rows within a subject are dependent.
4. Risk error rate by `test_time` quartile. Subgroups (sex, age tertile, baseline-severity tertile; cut points from subject-level values only)
   are computed only when a covariate CSV is supplied **and reproduces the saved rows**, and only above 8 clusters and 20 rows per class.
   Not implemented: error versus distance from each patient's threshold crossing (listed as a hypothesis below, not an output).
5. Disease: recording-level and subject-level blocks separately; leave-subject-out predictions are never pooled as if scores were on one scale without saying so.
6. Paired comparison of each candidate against the registered comparator on the same rows.

## 2. Failure modes expected from the design (hypotheses, not findings)

| Hypothesis | Why | How the output will show it |
|---|---|---|
| Risk errors concentrate near the threshold crossing of each patient | the label is a thresholded, interpolated, near-monotone trajectory | error by `test_time` quartile (distance-to-crossing analysis is not implemented) |
| Some patients are almost always positive or always negative | a few patients dominate the row count | per-subject accuracy spread; `subjects_with_both_classes` from the loader |
| Disease false positives come from few healthy subjects | only 8 healthy subjects | subject-level counts of misclassified subjects, with the 8-subject denominator |
| Scores are not calibrated | class re-weighting shifts scores | calibration slope and intercept with intervals |
| Voice adds little beyond time and baseline | label definition contains baseline UPDRS | paired AUC difference interval |

## 3. Status of each output

| Output | Status |
|---|---|
| `ERROR_ANALYSIS.generated.md` | not generated (`phase6_NOT_RUN.json` instead) |
| Subgroup tables | `NOT_RUN` |
| Per-subject profiles | `NOT_RUN` |
| Prediction artifacts (`protected/`, patient-level derivatives) | not produced; git-ignored when produced |
