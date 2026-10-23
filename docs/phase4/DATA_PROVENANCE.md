# NEURO-MAS Phase 4A: Data Provenance

**Read this first.** The two public datasets the shipped models were trained on **could not be obtained in this environment**.
No model figure in this repository has been re-computed from the underlying data in Phase 4A. This document records what is known
about the data, from what source, how well that is established, and exactly how to supply the files so the registered validation can run.

| Question | Answer |
|---|---|
| Were the original data files available to Phase 4A? | **No.** Both downloads fail with `Tunnel connection failed: 403 Forbidden` (`backend/ml/validation/results/fetch_FAILED.json`; reconfirmed this phase). No mirror or other legitimate source was reachable. Per the proxy policy the 403 was not retried or circumvented. |
| Is there a dataset hash lock? | **No.** `backend/ml/data/DATASETS.lock.json` does not exist: the download that would create it has never succeeded. |
| Were any figures reproduced from data? | **No.** Figures were checked for *internal consistency* against the shipped artifacts only (see "What was verified", below). |
| Is synthetic data ever labelled real? | **No.** Every synthetic fixture is tagged `SYNTHETIC_FIXTURE_NOT_EVIDENCE`; the runner exits with code 2 and writes a `*_NOT_RUN.json` record when no file is found; and there is an automated test for each of these behaviours. |

## 1. The two data sets

| | Disease classifier | Risk classifier |
|---|---|---|
| Source | UCI Machine Learning Repository, *Parkinsons* (dataset id 174), https://archive.ics.uci.edu/dataset/174, DOI 10.24432/C59C74 | UCI, *Parkinsons Telemonitoring* (id 189), https://archive.ics.uci.edu/dataset/189, DOI 10.24432/C5ZS3N |
| File | `parkinsons.data` | `parkinsons_updrs.data` |
| Licence (per publisher page) | CC BY 4.0 | CC BY 4.0 |
| Unit of observation | one sustained-vowel voice recording (22 acoustic features) | one telemonitoring session (voice measures plus clinician UPDRS scores) |
| Independent unit | the **subject** (parsed from `phon_R01_S01_1` style names) | the **patient** (`subject#`) |
| Observations used by the shipped model | 195 (artifact scaler: `n_samples_seen_ = 195`, matches metadata) | 5,801 (artifact scaler: `n_samples_seen_ = 5801`, matches metadata) |
| Subjects | metadata says **32**; publisher page says **31** (23 with PD) | 42 (metadata and publisher agree) |
| Rows per subject | about 6 | about 140 |
| Label | `status`: 0 healthy, 1 PD, constant within a subject | `worsening` = `total_UPDRS(t) − total_UPDRS(first visit) ≥ 3.0` (see 3) |
| Class balance (used rows) | 147 PD / 48 healthy; PD prevalence 0.7538; majority-class accuracy **0.7538** | 2,215 positive / 3,586 negative; positive rate 0.3818; majority-class accuracy **0.6182** |
| Missingness | publisher states none; **not verified** (file not available) | publisher states none; **not verified** |

Publisher facts above were read on 2026-10-23 from the dataset pages and were **summarised by a tool; they are not a hash-verified
download of the data**. Quoted figures should be re-checked against the pages. Where the repository's own model metadata and the
publisher disagree, both are given, because the disagreement itself is unresolved.

## 2. Which reported figures belong to which model

The Phase 4A brief refers to "42 subjects, pooled AUC 0.640, accuracy 0.630 vs majority baseline 0.618". Those are the **risk
classifier's** figures (Telemonitoring data, 42 patients), not the disease classifier's. They match the shipped risk metadata exactly:
accuracy 0.6304 (3,657/5,801), majority-class accuracy 0.6182 (3,586/5,801), pooled ROC-AUC 0.6405. The disease classifier's
shipped figures are different: pooled accuracy 0.7744 vs majority 0.7538, ROC-AUC 0.7941, from 195 recordings of 32 (or 31) subjects.

## 3. Label definitions

* **Disease:** `status` as published (0 healthy, 1 Parkinson's disease). A recording-level classification of who has PD; it says nothing about severity or progression. Because there are only 8 healthy subjects, all subject-level specificity statements rest on 8 units.
* **Risk:** `worsening = 1` when the clinician total UPDRS at a session exceeds that patient's first-visit total UPDRS by at least 3.0 points. The 3-point threshold is a **provisional project choice, not a validated minimal clinically important difference**. The label is **concurrent** with the voice features (same session), so the task is "does today's voice look like a worsened patient", not forecasting.
* **Interpolated labels.** The publisher states that `total_UPDRS` and `motor_UPDRS` are *linearly interpolated* between clinic assessments. Within a patient the label is therefore a smooth function of time, and the effective number of independent clinical measurements is far smaller than 5,801 rows (roughly 42 patients times a few assessments). Row-level counts overstate the information.
* **`test_time`.** In the dataset it is the time since **recruitment**. The API schema previously described the model input as time since baseline; the schema text has been corrected in Phase 4A (defect E02). How large the resulting train/serve mismatch is has not been quantified.

## 4. What was verified (from the shipped artifacts, with no data)

All from `backend/ml/validation/results/phase4a/artifact_audit.json`, regenerated by `python -m ml.validation.phase4a artifacts`.

| Check | Result |
|---|---|
| Model files' SHA-256 equal `MANIFEST.json` | Yes, both (the loader refuses to load otherwise) |
| Scaler row counts equal metadata row counts | Yes: 195 and 5,801 |
| Metadata accuracy/precision/recall + class counts reconstruct to **integer** confusion matrices | Yes, for all 5 disease candidates and both risk candidates |
| Winner accuracy minus majority accuracy | Disease **+0.0205** (4 of 195 recordings); risk **+0.0122** (71 of 5,801 rows) |
| Disease winner (`RandomForest_Top10Features`) recording-level confusion | TP 130, FN 17, FP 27, TN 21; sensitivity 0.884, **specificity 0.4375** |
| Risk winner (`LogisticRegression`) confusion | TP 1,325, FN 890, FP 1,254, TN 2,332; sensitivity 0.598, specificity 0.650 |
| Risk rows after dropping each patient's first visit | 5,801 stored vs 5,875 − 42 = **5,833 expected; 32 further rows unexplained** |
| Subject count, disease | metadata 32 vs publisher 31 |
| Risk coefficients | the five jitter deltas are collinear, with coefficients from −1.71 to +1.16 of mixed sign, so individual coefficients are not interpretable |
| Model metadata records dataset hash, library version, or seed | **No** to all three. `MANIFEST.json` states `trained_with_sklearn: 1.9.0`, inferred from the `InconsistentVersionWarning` raised when 1.9.1 unpickles the files, not recorded at training time. Loaded here under 1.9.1 |

"Reconstructs to integer confusion matrices" shows the metadata is internally consistent. It does **not** show the numbers are correct
for the real data, and it does **not** show that the evaluation behind them was leakage-free.

## 5. Dataset authentication (a MATCH is not clinical validity)

A SHA-256 lock would only prove a file is unchanged since first use, and no lock exists. Instead each shipped artifact contains a
`StandardScaler` fitted on its training matrix; its `n_samples_seen_`, `mean_` and `scale_` fingerprint that matrix.
`ml/validation/fingerprint.py::authenticate` rebuilds the matrix from a candidate file with the project's own feature construction and compares
(tolerances: mean difference ≤ 1e-6 standard deviations, scale ratio within 1e-6).

| Status | Meaning |
|---|---|
| `AUTHENTICATED_AS_TRAINING_DATA` | the file reproduces the matrix the shipped model was trained on |
| `NOT_THE_TRAINING_DATA` | a different or altered file; may be legitimate but is **not** a reproduction of the reported figures |
| `NOT_COMPARABLE` | unreadable, wrong columns, or any error (fails closed) |

A match says nothing about whether the labels are right (labels are not part of the fingerprint) or whether the data are clinically meaningful.
This closed a Phase 2 gap (defect E03): a synthetic file with the right shape would have been reported as `REAL_DATA_RUN`.

## 6. How to supply the data and run

The files must be downloaded on a machine with internet access and copied here (or referenced by path):

```
curl -fLo parkinsons.data         https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
curl -fLo parkinsons_updrs.data   https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data
sha256sum parkinsons.data parkinsons_updrs.data        # record these; this is the first hash lock the project will have
```

Path resolution order: `--csv`, then environment variable (`NEUROMAS_PARKINSONS_CSV`, `NEUROMAS_PARKINSONS_UPDRS_CSV`), then
`backend/ml/data/cache/<file name>`. If none exists the run stops with exit code 2 and writes `<kind>_NOT_RUN.json`.

```
cd backend
PYTHONPATH=. python -m ml.validation.phase4a authenticate --kind disease --csv /path/parkinsons.data          # exit 3 on mismatch
PYTHONPATH=. python -m ml.validation.phase4a authenticate --kind risk    --csv /path/parkinsons_updrs.data
PYTHONPATH=. python -m ml.validation.phase4a run --kind disease --csv /path/parkinsons.data
PYTHONPATH=. python -m ml.validation.phase4a run --kind risk    --csv /path/parkinsons_updrs.data
```

Environment of this phase: Python 3.13.17, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0,
Linux 6.18 x86_64. The protocol (id `p4a-1`, SHA-256 `0b2f10a7ec885419d9a2e72ad80b9d4c60b76987ea71e654edcb157b86723e42`) fixes seeds
(base 20261023), thresholds and feature sets before any real result exists; every output carries this hash.

## 7. Unresolved provenance issues

1. The original data have not been seen by this project's tooling in any phase; no hash is on record.
2. 32 risk rows are unexplained (5,833 expected, 5,801 used). Candidate causes (filtering, duplicates, a different file version) cannot be tested without the file.
3. 31 vs 32 subjects in the disease data.
4. The seed and dataset hash used to produce the shipped artifacts are not recorded, and the library version (1.9.0) is only inferred; the artifacts load under 1.9.1 with a version warning.
5. Publisher facts in section 1 are tool-summarised, not hash-verified.
