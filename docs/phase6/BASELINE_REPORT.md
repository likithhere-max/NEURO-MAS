# Phase 6: Reproducible Baseline Report

**Status: `NOT_RUN` (exit code 2).** The baseline cannot be reproduced because neither dataset file is available
(`DATA_AUTHENTICATION.md`, dependency D1). No performance number appears in this document because none was produced.
The historical figures below are retained and labelled; they are `HISTORICAL_UNVERIFIED_NOT_REPRODUCED`.
No clinical validity is claimed.

## 1. Starting state (verified at the start of Phase 6 from the repository)

| Item | Value | How verified |
|---|---|---|
| Branch | `phase2/advanced-ml-hardening` | `git` |
| Phase 5 head | `33bf7ba` on `8e4f4d3` (Phase 4A) | `git log` |
| Frozen validation protocol | `p4a-1`, SHA-256 `0b2f10a7ec885419d9a2e72ad80b9d4c60b76987ea71e654edcb157b86723e42` | pinned in `tests/phase5/test_p5_protocol_immutability.py` and `tests/phase6/test_p6_candidates_and_protocol.py` |
| Historical test baseline | 764 passed / 26 skipped (Phase 4A); 864 passed / 26 skipped (Phase 5) | recorded runs; Phase 6 run in `FINAL_READINESS_REPORT.md` |
| GitHub push | 403, not authorised, not retried | `FINAL_READINESS_REPORT.md` section 8 |

## 2. Historical figures (not reproduced)

| Figure | Value | Status |
|---|---|---|
| Disease pooled LOSO accuracy / AUC | 0.7744 / 0.7941 | `HISTORICAL_UNVERIFIED_NOT_REPRODUCED`; internally consistent only |
| Disease majority baseline | 0.7538 (147/195) | arithmetic on recorded class counts |
| Risk accuracy / AUC | 0.6304 / 0.6405 | `HISTORICAL_UNVERIFIED_NOT_REPRODUCED` |
| Risk majority baseline | 0.6182 (3,586/5,801) | arithmetic on recorded class counts; the brief's 0.6178 has no traceable source and is not used |

Gain over majority class is +0.0205 (4 of 195 recordings) and +0.0122 (71 of 5,801 rows) respectively; arithmetic only, no interval.
Both historical evaluations selected their reported model on the folds that produced the reported metric (Phase 5 finding F5).

## 3. Commands that produce the baseline (when data exist)

Registered protocol `p4a-1`, unchanged:

```
cd backend
PYTHONPATH=. python -m ml.validation.workflow run \
    --disease-csv /path/to/parkinsons.data --risk-csv /path/to/parkinsons_updrs.data \
    --out ml/validation/results/workflow
```

Recomputed comprehensive metrics, paired comparisons, calibration, decision curves and error analysis from the saved held-out
predictions (`MODEL_COMPARISON.md`, `ERROR_ANALYSIS.md`):

```
PYTHONPATH=. python -m ml.validation.phase6_report \
    --p4a-dir ml/validation/results/workflow --risk-csv /path/to/parkinsons_updrs.data \
    --out ml/validation/results/phase6/report
```

## 4. What was run in Phase 6 without data

| Run | Command (abridged) | Outcome | Committed record |
|---|---|---|---|
| Registered workflow | `workflow run --out .../phase6/baseline_no_data` | `NOT_RUN`, exit 2 | `backend/ml/validation/results/phase6/baseline_no_data/` |
| Prespecified extension | `workflow run --candidate-set p6-1-ext --out .../extension_no_data` | `NOT_RUN`, exit 2 | `.../extension_no_data/` |
| Report pipeline | `phase6_report --p4a-dir .../baseline_no_data --out .../report_no_data` | `NOT_RUN`, exit 2 (`phase6_NOT_RUN.json`) | `.../report_no_data/` |

All three manifests record commit `6a9fdf6` and `git_dirty: false`. They record sandbox paths and interpreter; those are
provenance, not configuration.

## 5. Environment (as run)

Python 3.13.17, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0. The shipped artifacts were pickled
under scikit-learn 1.9.0 and raise `InconsistentVersionWarning` on load; this is counted, not suppressed, and does not change
the artifacts' checksums.

## 6. Unresolved data dependencies

See `DATA_AUTHENTICATION.md` section 4 (D1-D10). Until D1 is resolved, every real-data row of every table in Phase 6 reads
`NOT_RUN`.
