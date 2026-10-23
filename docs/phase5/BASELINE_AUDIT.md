# Phase 5: Baseline Audit

Scope: what exists at commit `8e4f4d3` (end of Phase 4A), what each reported figure rests on, and which claims **cannot** be
verified in this environment. Nothing in this document is clinical validation. Historical reports are not rewritten; this file
links to them.

Historical reports this audit builds on (unchanged): `docs/phase4/ML_VALIDATION_REPORT.md`, `docs/phase4/DATA_PROVENANCE.md`,
`docs/phase4/BASELINE_COMPARISON.md`, `docs/audit/`, `docs/phase3/`.

## 1. Baseline test state (historical evidence, preserved)

| Run | Result | Where |
|---|---|---|
| Phase 4A report (historical) | **764 passed, 26 skipped** | `docs/phase4/ML_VALIDATION_REPORT.md` |
| Re-run of a clean detached worktree of `8e4f4d3` for this audit | **764 passed, 26 skipped, 66 warnings, 183 s** | scratchpad run, reproduced with `python -m pytest -q -p no:cacheprovider` from `backend/` |

The two runs agree. The 764/26 figure is retained as the historical baseline; Phase 5 results are recorded separately in
`FINAL_READINESS_REPORT.md`. The 26 skips are the live-service tests: 22 in `tests/rls_live/test_rls_matrix.py` (no separate Supabase test project configured)
and 4 in `tests/test_api.py` (live Supabase/Gemini tests disabled). They are unrelated to the ML data. Notably, **no pytest test depends
on the real UCI files**, so a green pytest run says nothing about real-data validation; that is why the workflow (section 7) is a separate
command with its own exit code. Phase 5 introduced no skip.

## 2. Model inventory

| | Disease classifier | Risk classifier |
|---|---|---|
| Artifact | `backend/ml/models/parkinsons_demo_classifier.joblib` | `backend/ml/models/parkinsons_risk_classifier.joblib` |
| SHA-256 matches `MANIFEST.json` | yes (verified by the audit and by the safe loader) | yes |
| Pipeline | StandardScaler, SelectKBest(f_classif, k=10), RandomForest(200, balanced, seed 42) | StandardScaler, LogisticRegression (C = 0.5) |
| Features | 22 acoustic features | 16 voice deltas + `test_time` + `baseline_total_UPDRS` = 18 |
| Training rows (scaler `n_samples_seen_`) | 195 | 5,801 |
| Subjects | metadata **32**, publisher **31** | 42 |
| Positives | 147 PD / 48 healthy (0.7538) | 2,215 / 5,801 (0.3818) |
| Label | `status`, group-level | `total_UPDRS(t) − total_UPDRS(first visit) ≥ 3.0`, concurrent, provisional |
| Records sklearn version / dataset hash / seed in metadata | **no / no / no** (before Phase 5; see `DATA_INTEGRITY_AUDIT.md` for the provenance sidecars added since) | **no / no / no** |

## 3. Data lineage

| Link | Status |
|---|---|
| Publisher | UCI ML Repository, ids 174 and 189 (CC BY 4.0 per publisher pages, summarised by a tool on 2026-10-23, not a hash-verified download) |
| Download in this environment | **Fails**: `403 Forbidden` from the egress proxy. No mirror was used. |
| Dataset hash lock | **None** existed. `backend/ml/data/DATASET_MANIFEST.json` (Phase 5) records the expected files with `UNPINNED` hashes; it is pinned only after a file is authenticated against the shipped artifact. |
| Training script | `ml/training/train_parkinsons_classifier.py`, `ml/training/train_risk_classifier.py` (the latter superseded by `ml/validation` + `train_final.py`) |
| Trained under | scikit-learn 1.9.0 (inferred from the unpickle version warning), not recorded by the artifact itself |

## 4. Where each headline figure comes from

| Figure | Source | Reproduced from data? |
|---|---|---|
| Disease pooled LOSO accuracy 0.7744, AUC 0.7941 | shipped metadata | **No.** Internally consistent only. |
| Disease confusion TP 130 / FN 17 / FP 27 / TN 21 | reconstructed from metadata accuracy, precision, recall and class counts; integers, consistent | **No.** |
| Disease specificity 0.4375 | derived from the confusion matrix above | **No.** |
| Risk accuracy 0.6304, AUC 0.6405 | shipped metadata | **No.** |
| Risk confusion TP 1,325 / FN 890 / FP 1,254 / TN 2,332 | reconstructed, integer-consistent | **No.** |
| Majority-class accuracy 0.7538 (disease), 0.6182 (risk) | arithmetic on the class counts the artifacts record (147/195, 3,586/5,801) | arithmetic only |
| Gain over majority: +0.0205 (4 of 195 recordings), +0.0122 (71 of 5,801 rows) | arithmetic | arithmetic only |

"Internally consistent" means the integers and ratios agree with each other and with the scaler row counts. It does **not**
mean the numbers were produced by a leakage-free evaluation of the real data; the metrics come from the original training
scripts, whose protocol is audited in `VALIDATION_PROTOCOL_AUDIT.md`.

### The 0.6178 figure

The Phase 4B/5 brief quotes a risk majority baseline of **0.6178**. That number does not appear anywhere in the repository
(code, metadata, JSON or documents). The traceable value is **0.6182** (3,586 / 5,801). The two differ by 0.0004. No source for 0.6178 could be found, so this audit does not guess how it arose; it is recorded as **unverified** and not used. The tests check 0.6182 against the
recomputed artifact audit.

## 5. Historical evaluation design (disease), as found in the code

`ml/training/train_parkinsons_classifier.py` builds, per candidate, a `Pipeline(scaler, [SelectKBest(f_classif, k=10)], classifier)`
and scores it with `LeaveOneGroupOut` over the parsed subject. Consequences:

* Scaling and feature selection are **inside** each LOSO fold. There is no direct feature-selection leakage in the reported number.
* The reported winner was **chosen from five candidates on the same folds that produced the reported metric**. That is
  optimistic selection bias that the metadata does not quantify; Protocol `p4a-1` replaces it with nested subject-grouped
  selection.
* Pooled LOSO predictions pool scores from 32 different models; the pooled AUC mixes score scales.
* Only 8 healthy subjects exist. Specificity is based on 8 independent units, however many recordings they contribute.

## 6. Claims that cannot be verified here

1. That any historical metric is reproducible from the public files.
2. That the risk training set is 5,801 rows for a reason other than the tie-dropping rule (see `DATA_INTEGRITY_AUDIT.md`).
3. The subject count of the disease file (31 vs 32).
4. The size of the training/serving `test_time` skew.
5. Whether voice-delta features add information beyond `test_time` and baseline UPDRS.
6. Anything about external, prospective or clinical performance. There is no external cohort in this repository.
7. The 0.6178 majority figure in the brief.

## 7. How to reproduce (when authorized data is available)

```
cd backend
PYTHONPATH=. python -m ml.validation.dataset_manifest status
PYTHONPATH=. python -m ml.validation.workflow run --disease-csv /path/parkinsons.data --risk-csv /path/parkinsons_updrs.data --out ml/validation/results/workflow
```

Without the files the same command exits **2** (`NOT_RUN`) and writes `experiment_manifest.json` plus `*_NOT_RUN.json`.
