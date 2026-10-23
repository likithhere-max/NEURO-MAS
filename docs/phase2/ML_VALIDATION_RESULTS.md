# ML Validation Results (Phase 2, B)

## 0. Bottom line (read this first)
**No real-data performance result exists.** Both validation runs are **NOT RUN (BLOCKED)**: the datasets could not be
downloaded from this environment. No AUC, accuracy, sensitivity, specificity or calibration figure in this repository was
produced by the Phase 2 protocol. The numbers shipped in `ml/models/*_metadata.json` come from the original, optimistic
selection procedure (Phase 1 finding ML1) and remain **NOT_INDEPENDENTLY_VALIDATED**. Both models remain
**CLINICALLY UNVALIDATED** and are labelled demonstrations.

## 1. Status table
| Step | Command | Status | Evidence file |
|---|---|---|---|
| Download datasets | `validate_models fetch` | **FAILED** (re-attempted 2026-10-23: `URLError: Tunnel connection failed: 403 Forbidden`) | `ml/validation/results/fetch_FAILED.json` |
| Disease model, nested LOSO | `validate_models disease` | **NOT RUN** (no data) | — |
| Risk model, nested GroupKFold + ablations | `validate_models risk` | **NOT RUN** (no data) | — |
| Permutation tests, PPV/NPV, calibration | part of the two above | **NOT RUN** | — |
| Artifact integrity | `validate_models artifacts` | **COMPLETED (real run on the repo's files)** | `ml/validation/results/artifacts.json` |
| Code-path check of the runner on fake data | `validate_models synthetic-smoke --fast` | COMPLETED — output labelled `SYNTHETIC_FIXTURE_NOT_EVIDENCE`; **numbers deliberately not reproduced here** | (kept outside the repo) |
| Unit tests of the evaluation kit and runner | `pytest tests/test_ml_evalkit.py tests/test_ml_validation_runner.py` | COMPLETED (all pass; see INTEGRATION_TEST_REPORT.md) | — |

## 2. What the real artifact run found (2026-10-23, scikit-learn 1.9.1)
| Artifact | SHA-256 matches `MANIFEST.json` | Feature count matches metadata | Loads without warnings |
|---|---|---|---|
| `parkinsons_demo_classifier.joblib` | yes | yes | **no** — "estimator … from version 1.9.0 when using version 1.9.1 … may lead to breaking code or invalid results" (Pipeline, StandardScaler, SelectKBest, RandomForest, DecisionTree) |
| `parkinsons_risk_classifier.joblib` | yes | yes | **no** — same warning (Pipeline, StandardScaler, LogisticRegression) |

Also: neither metadata file records the scikit-learn version or the training-dataset hash that produced it
(`metadata_records_sklearn_version: false`, `metadata_records_dataset_hash: false`), so provenance of the shipped numbers
cannot be reconstructed from the repo. **Recommendation:** after obtaining the data, re-train under the pinned
scikit-learn (`requirements.lock.txt`: 1.9.1), regenerate the manifest and write version + data hash into the metadata.

## 3. Real-data results
*Intentionally empty.* Paste the contents of `results/disease.json` and `results/risk.json` here after running §10 of
ML_EXPERIMENT_PROTOCOL.md; do not summarise them with a single headline number. Every table must show the cluster-bootstrap CI, the
no-skill and classical baselines, the ablations and the PPV/NPV-at-prevalence table.

## 4. What the earlier (Phase 1) analysis says, for context only
Reading the training code and metadata (not re-measuring): model selection and scoring used the same folds (optimistic);
the disease file has ~31–32 people and ~75 % PD, so any single number rests on a handful of healthy subjects; the risk model
uses `test_time` and `baseline_total_UPDRS` as inputs and its label is "UPDRS rose since first visit", so those two features
may explain much of its signal without any voice information (an untested hypothesis — the ablation exists to test it).

## 5. Guards that exist regardless of validation
* Inputs outside the training domain or violating physical/algebraic constraints produce an **abstention** (no label, no probability), and the UI renders it as "No prediction made".
* Artifacts are loaded only after a SHA-256 check and through a restricted unpickler.
* Responses and UI carry the demonstration disclaimer; nothing from these models enters the clinical workflow or the consensus.
