# Phase 5: Data Integrity Audit

Four open integrity problems were carried out of Phase 4A. This document states, for each, what was established from code and
tests, what only the real data can settle, and what was added so that the data settles it automatically the moment it is
supplied. **No count was forced, and no number below comes from the real data files, which are not available here.**

| # | Problem | Status |
|---|---|---|
| 1 | Risk: 5,801 rows used vs 5,833 expected | Mechanism identified in code and demonstrated by test; actual tie count **needs the data** |
| 2 | Disease: 32 subjects (metadata) vs 31 (publisher) | Parser eliminated as the cause; actual cause **needs the data** |
| 3 | `test_time` convention mismatch between training and serving text | **Resolved in code**: one convention, enforced by tests. Size of any real skew **needs the data** |
| 4 | Artifact metadata lacks dataset hash, seed, library versions | **Resolved for all new artifacts**; shipped artifacts honestly declared `INCOMPLETE_DECLARED` |

## 1. Risk rows: 5,801 vs 5,833

Arithmetic: 5,875 raw rows, 42 patients. If only each patient's first visit were removed, 5,833 rows would remain. The shipped
scaler saw 5,801. The gap is **32 rows**.

Root cause in the code (`ml/validation/data.py::build_risk_frame`, and identically in the superseded
`ml/training/train_risk_classifier.py`): the baseline row is identified by `is_first = test_time == groupby(subject).min()`.
That drops **every** row tied at a patient's minimum `test_time`, not one row per patient. Each tie beyond the first costs one
additional row, so 32 such ties would produce exactly 5,801.

What is and is not established:

* **Established by test** (`test_mechanism_32_ties_turn_5875_rows_into_5801_which_is_the_shipped_count`): on a constructed frame
  of 42 subjects and 5,875 rows with 32 ties, the rule yields 5,801. This demonstrates the mechanism is *sufficient*. It does
  **not** assert the real file has 32 ties.
* **Not established**: the real tie count. Other causes (a different filter at training time, a different file revision) are
  not excluded by this evidence.
* `risk_exclusion_report(raw)` in `ml/validation/exclusions.py` produces the full accounting from the real file: raw rows,
  subjects, ties dropped, rows kept, kept minus expected, whether the accounting identity holds, subjects left with no
  training rows, data-quality counters, and a comparison with the shipped artifact's 5,801. If the file does not reproduce
  5,801 the report says so; it does not adjust anything (`test_a_file_that_does_not_reproduce_the_shipped_count_says_so`).
* `reconcile_risk_rows` asserts the report agrees with `build_risk_frame` row for row (tested on random data).

Design observation (a finding, not changed): the rule also removes legitimate repeat measurements at the first timestamp and
defines the baseline row as the first in file order among ties (`test_shipped_baseline_row_is_the_first_in_file_order_among_ties`).
Changing the rule would change the estimand and the shipped numbers, so it is documented, not altered. If the real data show
ties, the choice between "drop all ties" and "drop one per patient" should be made in a **versioned protocol amendment** before
any confirmatory run.

## 2. Disease subjects: 32 vs 31

The recording names look like `phon_R01_S01_1`. Two parsers exist in the repository (training script
`name.str.split("_").str[2]`, and the Phase 2 loader). Tests show they agree on every well-formed name and differ only on
malformed names, which `disease_subject_report` surfaces rather than hides. **The parser is therefore not the cause.**

What remains possible, and only the file can tell: the shipped metadata's 32 may count a recording-name group that the
publisher merges or omits; the publisher's text says 31 subjects (23 with PD) while its own instance-count metadata field says
197 against the 195 recordings in the text. `disease_subject_report(raw)` lists the subject codes, recordings per subject and
labels per subject, and **refuses** any subject with two labels (`test_subject_with_two_labels_is_refused`). Nothing assumes 31
or 32.

Why it matters: 24 PD / 8 healthy (metadata) versus 23 PD / 8 healthy (publisher) changes the number of independent units on
which every subject-level statistic rests (see `STATISTICAL_EVIDENCE_REPORT.md`).

## 3. `test_time` convention

Publisher definition (page read 2026-10-23, tool-summarised): time since **recruitment**, integer part = days since recruitment.
Before Phase 5 the API schema described it as time since baseline (defect E02, text corrected in 4A), and nothing enforced
a single meaning.

Phase 5 changes:

* `app/ml/contract.py` holds the single convention (`days_since_recruitment_v1`) and its description. The request schema and the
  provenance contract both import it.
* The out-of-distribution bound for `test_time` is `lo = 0.0` (negative values abstain). The comment states this is a
  conservative engineering bound, not a clinical rule.
* `test_no_serving_or_ml_module_still_calls_test_time_days_since_baseline` scans the source so the old wording cannot return.
* The convention id is written into every provenance record.

Training/serving parity (`test_every_training_row_maps_to_the_identical_vector_through_the_serving_path`,
`test_serving_probability_equals_direct_model_probability_for_the_same_valid_input`): all 18 features map to the same vector
through the schema aliases and the serving path as through the training path, and the served probability equals the model's
direct probability.

Open and data-dependent: whether the shipped model's `test_time` distribution matches recruitment-relative days. The sidecar
records the scaler mean and scale for `test_time`; the workflow compares a supplied file to them (authentication). Until then
no claim about skew magnitude is made.

Input handling tested: NaN/Inf rejected with 422, out-of-range values abstain, missing or mistyped fields rejected (never
defaulted), model unavailable returns 503 (never a default answer), duplicate requests are deterministic.

## 4. Artifact provenance

Required fields and statuses (`ml/validation/artifact_contract.py`): `COMPLETE`, `INCOMPLETE_DECLARED`, `INVALID`.

* Shipped artifacts cannot be backfilled truthfully (no dataset hash, no seed, no library version were recorded), so
  `parkinsons_*_provenance.json` sidecars were generated with those fields declared unrecoverable. They report
  `INCOMPLETE_DECLARED` and **never** `COMPLETE`. A test asserts the committed sidecars equal what the generator produces from
  the artifact; another asserts a missing field invalidates a record and a malformed checksum is invalid.
* `train_final.py` now writes a `COMPLETE` record for newly trained artifacts (dataset hash, seed, library versions, feature
  units taken only from feature names, training-matrix moments, `test_time` convention, target definition).
* `GET /api/v1/ml-demo/*/info` exposes the provenance status.
* Defect E05 (found by testing): a retrained pipeline with different step names, or without a scaler, broke fingerprint
  authentication and the serving input domain. Fixed by finding the scaler by type, falling back to sidecar moments, and
  returning 503 when no input domain can be derived.
* Defect E06 (found by testing): the restricted unpickler refused `HistGradientBoostingClassifier` pipelines. Three exact
  names were added to the allow-list; tests confirm that other names (`BitGenerator`, `PCG64DXSM`, `numpy.load`, `os.system`,
  `builtins.eval`) are still refused. This slightly widens the trusted-class surface and is reviewable in `app/ml/safe_load.py`.

## 5. Dataset manifest

`backend/ml/data/DATASET_MANIFEST.json` lists both files with `UNPINNED` hashes. `dataset_manifest.py` supports
`status|verify|pin`. `pin` requires structural verification **and** authentication against the shipped artifact, so a hash cannot
be pinned to a file that is not the training data. Statuses: `PINNED_MATCH`, `UNPINNED`, `HASH_MISMATCH`, `MISSING_FILE`;
an unpinned manifest never returns a success status. `status` currently exits 2.
