# Phase 6: Data Authentication

**Status: `BLOCKED` on data access.** No dataset file has been retrieved by this project, in any phase. Nothing below was
authenticated against a real file. This document records what authentication *is*, how it is enforced in code, what was
attempted, and the exact list of unresolved dependencies. No clinical validity is claimed.

## 1. What "authenticated" means here

A user-supplied CSV is accepted as a reproduction input only if all of the following hold, in this order. Each step refuses
or downgrades the evidence label; none can be skipped by a flag.

| Step | Check | Failure behaviour | Code |
|---|---|---|---|
| 1 | File exists at a documented path or `--csv` / environment variable | `NOT_RUN`, exit 2, `*_NOT_RUN.json` | `ml/validation/workflow.py` |
| 2 | Dataset manifest hash, if pinned | **refuse** (exit 3) on mismatch | `ml/validation/dataset_manifest.py` |
| 3 | Fingerprint against the **shipped artifact's** scaler moments (`n_samples_seen_`, mean, scale) | **refuse** (exit 3) unless `AUTHENTICATED` | `ml/validation/fingerprint.py` |
| 4 | Structural integrity: row count, class counts, column set, no missing values | **refuse** (exit 3) | `ml/validation/data.py` loaders |
| 5 | Data-quality and exclusion reports written; evidence label set (`+MANIFEST_UNPINNED` while no pin exists) | label only | `ml/validation/exclusions.py` |
| 6 | Human pin (`dataset_manifest pin`) records SHA-256 and byte size | only after step 3 passes | `ml/validation/dataset_manifest.py` |

An unauthenticated file does **not** run through the registered workflow. The only route for one is the separate exploratory CLI
`ml.validation.phase4a`, which labels the output `UNAUTHENTICATED_USER_FILE_NOT_A_REPRODUCTION_OF_THE_REPORTED_FIGURES`.

**Correction to Phase 5 text.** `docs/phase5/REPRODUCIBILITY.md` section 4 says an unauthenticated file "still runs but is labelled".
That is true of `phase4a`, not of `workflow run`, which refuses (verified in `workflow._experiment`). The Phase 5 file is left as
written (append-only history); this document is the corrected statement (finding F8 in `FINAL_READINESS_REPORT.md`).

Step 3 is the strongest available check: the shipped scalers were fitted on the original training data, so a file that
reproduces their means and scales to numerical tolerance is, with overwhelming probability, the data the models saw. It
authenticates against the *artifact*, not against the publisher. Publisher identity (UCI landing page and DOI) is recorded in
the manifest but a hash that originates from the publisher is not available to this project.

## 2. Current state (manifest `1.1.0`)

| | `parkinsons.data` (UCI 174) | `parkinsons_updrs.data` (UCI 189) |
|---|---|---|
| Retrieved | **no** | **no** |
| SHA-256 / bytes | `null` / `null` (not honestly computable) | `null` / `null` |
| Pin status | `NOT_PINNED` | `NOT_PINNED` |
| Fingerprint authentication | `NOT_RUN` | `NOT_RUN` |
| Expected structure (code constants, **unverified against a file**) | 195 rows, 147 PD / 48 healthy, 24 columns | 5,875 rows, 42 patients; 5,801 used by the shipped model |
| Publisher-stated facts (not hash-verified) | 31 subjects | 42 patients |
| Shipped-artifact facts | 195 rows, **32** metadata subjects | 5,801 rows |

`overall_status` is `UNPINNED_NO_ACCESS`; `python -m ml.validation.dataset_manifest status` exits 2.

## 3. Access attempts (all recorded, none retried against policy)

| Phase | Attempt | Outcome |
|---|---|---|
| 2 | urllib download through the session proxy | HTTP 403 (`ml/validation/results/fetch_FAILED.json`, retained) |
| 4A | re-confirmed; GitHub access is bound to configured repositories; no authorised mirror found | not retrieved |
| 5 | not retried: the proxy policy states a 403 denial is not to be retried or worked around | not retrieved |
| 6 | (a) proxy policy read; no retry. (b) request to read the user's Downloads/Documents folders for local copies | **denied**; no local copy examined. Not retried. |

No unverified mirror, no regenerated or "equivalent" dataset, and no synthetic substitute was used for any result in this
repository. Synthetic fixtures exist for engineering tests only and carry `SYNTHETIC_FIXTURE_NOT_EVIDENCE`.

## 4. Unresolved data dependencies

Each item names who can resolve it and what it unblocks.

| # | Dependency | Owner | Unblocks |
|---|---|---|---|
| D1 | Authorised copies of `parkinsons.data` (UCI 174) and `parkinsons_updrs.data` (UCI 189). Either attach them to this conversation, or run the commands in section 5 on a machine that can reach UCI. | user / any machine with access | every `NOT_RUN` experiment in `BASELINE_REPORT.md`, `MODEL_COMPARISON.md`, `ERROR_ANALYSIS.md`, and the voice-value study |
| D2 | If local copies exist: grant this session access to the folder holding them (the Phase 6 request was denied and was not repeated) | user | same as D1 |
| D3 | Pin hashes after fingerprint authentication (`dataset_manifest pin`) | human with the file | manifest status becomes `PINNED` |
| D4 | Settle the actual tie count behind 5,801 vs 5,875 (the tie-dropping rule is a sufficient mechanism, proven by test; the real count needs the file) | file | finding F1 closed or amended via `p4a-2.json` |
| D5 | Settle disease subject count 31 vs 32 (parser eliminated as cause in Phase 5) | file | finding F2 closed |
| D6 | Measure the `test_time` training/serving skew | file | hazard 8 in `SAFETY_CASE.md` quantified |
| D7 | Verify `expected_columns` against the real headers | file | manifest `expected_columns` upgraded from code-derived to observed |
| D8 | Independent external cohort with its own data-use approval | clinical / organisational | all external-validation claims (`CLINICAL_VALIDATION_PROTOCOL.md`) |
| D9 | Clinical decisions: intended use, endpoint, acceptable error trade-offs (`REQUIRES_CLINICAL_INPUT` in protocol `p6-1`) | qualified clinicians | any acceptance criterion |
| D10 | GitHub push (403, not authorised) | user | none of the science; local commits are the record until pushed |

## 5. Commands to resolve D1-D7 (authorised path only)

```
cd backend
PYTHONPATH=. python -m ml.validation.dataset_manifest status          # exits 2 until pinned
PYTHONPATH=. python -m ml.validation.workflow run \
    --disease-csv /path/to/parkinsons.data \
    --risk-csv    /path/to/parkinsons_updrs.data \
    --out ml/validation/results/workflow                              # registered p4a-1
PYTHONPATH=. python -m ml.validation.workflow run --candidate-set p6-1-ext \
    --disease-csv /path/to/parkinsons.data \
    --risk-csv    /path/to/parkinsons_updrs.data \
    --out ml/validation/results/workflow_p6ext                        # prespecified extension (exploratory)
PYTHONPATH=. python -m ml.validation.phase6_report \
    --p4a-dir ml/validation/results/workflow \
    --ext-dir ml/validation/results/workflow_p6ext \
    --risk-csv /path/to/parkinsons_updrs.data \
    --out ml/validation/results/phase6/report                         # recomputed metrics + paired comparisons
```

The extension run refuses to write into a directory holding a different candidate set's manifest (tested), so the two
output directories must differ. Prediction files contain patient-level derivatives and go to `protected/`, which is
git-ignored; do not commit them.

## 6. Verification performed in Phase 6 without data

* Dataset manifest `1.1.0` parses; `test_shipped_manifest_is_honestly_unpinned` (Phase 5) pins that no hash is recorded while no file was retrieved.
* Workflow, extension workflow and report pipeline each exit 2 with committed `NOT_RUN` records
  (`backend/ml/validation/results/phase6/{baseline,extension,report}_no_data/`), produced at a clean commit (`git_dirty: false`).
* The end-to-end report pipeline is tested on **forged** data matched to stand-in artifacts, to prove the code path and the
  fail-closed gates. Those outputs are `SYNTHETIC_FIXTURE_NOT_EVIDENCE` and are not committed.
