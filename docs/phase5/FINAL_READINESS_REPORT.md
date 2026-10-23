# Phase 5: Final Readiness Report

**Verdict.** The engineering needed to produce trustworthy evidence is in place and tested; the evidence itself has **not been
produced**, because neither public dataset can be downloaded in this environment (403 from the egress proxy, no authorised
mirror). Every real-data experiment is `NOT_RUN`. **No clinical validity is claimed, and none can be inferred from this
report.** Clinical readiness cannot be declared by engineering criteria; both models remain research demonstrations.

## 1. What was and was not established

| Question | Answer |
|---|---|
| Were the historical figures reproduced from data? | **No.** Data unavailable. Only internal consistency of shipped metadata was checked (unchanged from Phase 4A; the no-data audit re-run is identical apart from the environment block). |
| Did the registered protocol `p4a-1` run on real data? | **No.** `NOT_RUN`, exit code 2, manifest committed. |
| Was the protocol changed? | **No.** Hash `0b2f10a7...723e42` pinned in a test; no amendment was needed. |
| Is there external or prospective evidence? | **No.** None exists in the repository. |
| Is any synthetic result presented as real? | **No.** Synthetic fixtures are labelled `SYNTHETIC_FIXTURE_NOT_EVIDENCE`; planning output is labelled `DESIGN_STAGE_PLANNING_NOT_EVIDENCE`. |
| Does voice add information beyond time and baseline UPDRS? | **Unknown.** The design could detect roughly +0.08 AUC (optimistic planning figure) or more. |

## 2. Work by stage

| Stage | Outcome | Document |
|---|---|---|
| 1 Baseline audit | model inventory, lineage, provenance of each figure, list of unverifiable claims; clean-worktree baseline 764 passed / 26 skipped | `BASELINE_AUDIT.md` |
| 2 Data integrity | risk 5,801 vs 5,833: mechanism (tie-dropping rule) proven sufficient by test, real tie count needs data; disease 31 vs 32: parser eliminated; single `test_time` convention enforced; exclusion/subject/data-quality reports; dataset manifest | `DATA_INTEGRITY_AUDIT.md` |
| 3 Validation and baselines | protocol audited, immutability tests, leakage tests, limitations listed, no estimand-changing defect | `VALIDATION_PROTOCOL_AUDIT.md` |
| 4 Real-data experiments | **blocked**; `NOT_RUN` manifest committed | `ml/validation/results/phase5/workflow_no_data/` |
| 5 Generalization and robustness | engineering stress tests only, labelled; no external cohort | `GENERALIZATION_AND_ROBUSTNESS.md` |
| 6 Clinical evidence | intended-use statement (no thresholds invented), 21-row gap register | `CLINICAL_INTENDED_USE.md`, `CLINICAL_EVIDENCE_GAPS.md`, `SAFETY_AND_HUMAN_OVERSIGHT.md` |
| 7 Reproducibility | provenance contract, training/serving parity, one workflow, fail-closed CI semantics | `REPRODUCIBILITY.md` |
| 8 Final audit | this report; doc-number and no-claim tests | this file, `STATISTICAL_EVIDENCE_REPORT.md` |

Statistical planning (not evidence): `STATISTICAL_EVIDENCE_REPORT.md` and `backend/ml/validation/results/phase5/power_planning.json`.

## 3. Defects and findings

| ID | Finding | Resolution |
|---|---|---|
| F1 | Risk baseline rule drops **all** rows tied at a patient's minimum `test_time`, not one per patient; 32 ties would explain 5,801 | documented; mechanism test; actual count awaits data; rule change would need an amendment |
| F2 | Disease subject parsers agree on all well-formed names | parser excluded as cause of 31 vs 32 |
| F3 | `test_time` described inconsistently (baseline vs recruitment) | single convention `days_since_recruitment_v1`, scan test, lower bound 0 |
| F4 | Artifact metadata lacked dataset hash, seed, library version | provenance contract; new artifacts `COMPLETE`; shipped ones `INCOMPLETE_DECLARED` (cannot be backfilled truthfully) |
| E05 | Retrained pipelines with other step names or no scaler broke authentication and the serving input domain | scaler found by type, sidecar-moment fallback, 503 when no domain |
| E06 | Safe loader refused `HistGradientBoostingClassifier` pipelines | three exact allow-list additions; other names still refused (tested). Slightly widens the trusted-class surface; reviewable in `app/ml/safe_load.py` |
| F5 | Historical disease evaluation selected its winner on the same LOSO folds it reports (feature selection itself was in-fold) | documented; `p4a-1` uses nested selection |
| F6 | Brief's risk majority figure 0.6178 not found anywhere; traceable value is 0.6182 | recorded as unverified; test asserts it is absent from code and metadata |

## 4. Test results

| Run | Result |
|---|---|
| Historical (Phase 4A, `8e4f4d3`, clean worktree re-run) | 764 passed, 26 skipped |
| Phase 5 only (`tests/phase5`, first run, before the doc-number and dirty-check tests) | 92 passed, 0 skipped |
| **Phase 5 full backend suite** | **864 passed, 26 skipped, 79 warnings** (225.6 s) |

Difference from the historical run: +100 passing tests (the Phase 5 tests); **the 26 skips are the live-service tests** (22 RLS tests with no
Supabase test project, 4 live Supabase/Gemini API tests; listed with `-rs` in the Phase 5 run; the baseline run was made without `-rs`, so
its skip list was not printed, but the count is identical). No pytest test depends on the real UCI files. No test was weakened, no assertion removed,
no acceptance criterion changed, no warning suppressed.

Warnings rose from 66 to 79. By type, the full run shows: scikit-learn `InconsistentVersionWarning` (artifacts pickled under 1.9.0,
run under 1.9.1; present in the baseline too), a scikit-learn small-class `UserWarning` (present in the baseline), and two types
**not present in the baseline output**: a `DeprecationWarning` that the process is multi-threaded at `os.fork()` (raised in the
pre-existing `tests/test_math_properties.py::test_anomaly_simulation_is_seed_reproducible`, so it appears to be a side effect of
threads left by earlier tests, possibly the new ones; not investigated further), and a hypothesis-plugin notice about skipping
collection of a local `.hypothesis` directory (environmental). The per-instance count increase was not triaged beyond these types.

## 5. Commands

```
cd backend
python -m pytest -q -p no:cacheprovider                                   # 864 passed, 26 skipped
PYTHONPATH=. python -m ml.validation.dataset_manifest status             # exits 2 (unpinned)
PYTHONPATH=. python -m ml.validation.workflow run --out ml/validation/results/phase5/workflow_no_data   # exits 2 (NOT_RUN)
PYTHONPATH=. python -m ml.validation.phase4a artifacts --out <dir>       # deterministic no-data audit
PYTHONPATH=. python -m ml.validation.power_planning --reps 20            # planning only
```

With the data: `docs/phase5/REPRODUCIBILITY.md`, section 4.

## 6. Artifacts

| Artifact | Path |
|---|---|
| Experiment manifest (machine readable, NOT_RUN) | `backend/ml/validation/results/phase5/workflow_no_data/experiment_manifest.json` |
| NOT_RUN records and generated report | same directory |
| Model artifact metadata (provenance sidecars) | `backend/ml/models/parkinsons_demo_classifier_provenance.json`, `parkinsons_risk_classifier_provenance.json` |
| Dataset manifest | `backend/ml/data/DATASET_MANIFEST.json` |
| Frozen protocol | `backend/ml/validation/protocol_registry/p4a-1.json` |
| Planning output | `backend/ml/validation/results/phase5/power_planning.json` |

## 7. Changed files

New: `app/ml/contract.py`; `ml/validation/{exclusions,dataset_manifest,artifact_contract,report_from_artifacts,workflow,power_planning}.py`;
`ml/validation/protocol_registry/p4a-1.json`; `ml/data/DATASET_MANIFEST.json`; two provenance sidecars; `tests/phase5/*`
(seven test modules and helpers); `docs/phase5/*`.
Modified: `app/schemas/ml_demo.py`, `app/ml/ood.py`, `app/ml/safe_load.py`, `app/api/routes/ml_demo.py`,
`ml/validation/fingerprint.py`, `ml/validation/phase4a.py` (observe-only recorder), `ml/training/train_final.py`,
`tests/phase4/helpers_p4.py`, `.gitignore` (protected outputs), and append-only pointers in `docs/phase4/*.md`.
Not modified: shipped model files, `MANIFEST.json`, the protocol, historical results.

## 8. Remaining blockers

1. **Data access (external).** Authorised copies of UCI 174 and 189 are needed. Without them gaps 1-6 and 20-21 in `CLINICAL_EVIDENCE_GAPS.md` stay open.
2. **Independent cohort, ethics and clinical decisions (external).** External and prospective validation, intended use, endpoint definition and hazard analysis are clinical and organisational work.
3. **GitHub push (external).** Push is blocked (403) and not authorised; it was not retried or worked around. Phase 5 work is committed locally only and must be pushed from a machine with access.

## 9. Next actions, in order

1. On a machine with access, obtain both files, run `dataset_manifest status`, then the workflow (REPRODUCIBILITY section 4).
2. Read the exclusion and subject reports; settle F1 and the 31/32 question with real counts. If the tie rule should change, write `p4a-2.json` before any confirmatory run.
3. Read the registered result for the voice-value question and the baselines; report subject-level intervals, never row-level ones alone.
4. Pin the dataset hashes; retrain via `train_final.py` to obtain `COMPLETE` provenance if the artifacts are to be replaced.
5. Only then plan external validation with clinicians; keep both models out of any care pathway until the gaps are closed by qualified people.

## 10. Statement on claims

No document or test in this phase asserts clinical validity, external or prospective validation, calibration, fairness, or
clinical utility. `test_no_phase5_document_claims_clinical_validity` fails if such a claim is written without negation.
A reproducible negative result, a model that abstains, and an explicitly blocked validation are all preferred to an unsupported
success; this phase produced the third.

---
*Appended in Phase 6 (history above is unchanged).* Status after Phase 6: `docs/phase6/FINAL_READINESS_REPORT.md`. Real-data experiments are still `NOT_RUN`.
One statement in `REPRODUCIBILITY.md` section 4 is corrected in `docs/phase6/DATA_AUTHENTICATION.md`: the registered workflow refuses an unauthenticated file (exit 3); only `ml.validation.phase4a` runs one, labelled.
