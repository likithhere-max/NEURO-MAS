# Phase 6: Architecture, Test, CI and Research-Credibility Audit (workstream 8)

Scope: inspection of what exists, with findings and the small fixes made. No feature was added for its own sake. Facts below were read
from the repository at Phase 6; where a statement comes from an earlier audit it is attributed. No clinical validity is claimed.

## 1. Multi-agent architecture

| Observation | Source | Assessment |
|---|---|---|
| Ten agent modules (`copilot`, `data_intake`, `diagnostic`, `emergency`, `knowledge`, `longitudinal`, `medication`, `personalization`, `risk`, `signal_analysis`) behind a registry with declared dependencies | `backend/app/agents/` | Clear separation; each declares `reliability_weight` |
| Orchestration is an explicit async state machine over typed `WorkflowState` (not a graph framework); failed or timed-out agents mark the run `degraded` instead of passing silently | `app/orchestrator/orchestrator.py`; `docs/ARCHITECTURE.md` | Auditable, small (365 lines), concurrency and timeout tests exist |
| Reliability weights are informal constants (0.5 to 1.0) | `agents/*.py`; Phase 3 `AUDIT_REPORT.md` M3 | **Unvalidated.** Honest labelling exists; no calibration is possible without outcome data |
| "Consensus" is effectively the diagnostic agent's retrieval score today (other agents do not emit per-disease support) | Phase 3 `MATHEMATICAL_SPECIFICATION.md` | An aggregation framework, not evidence of multi-agent agreement. The UI text says so. Should not be cited as a research contribution |
| Emergency path is deterministic with weight 1.0 | `agents/emergency.py` | Appropriate for a safety path; its rules are not clinically validated |
| ML models (`app/ml`, `/ml` demo routes) are not consumed by any agent | Phase 5 source search | Keeps the hazard surface small (SAFETY_CASE C1) |

Research-credibility reading: the defensible contribution is the engineering of a fail-closed, auditable pipeline and a validation framework
that refuses to overstate; it is not predictive performance, which has not been demonstrated on authenticated data, and it is not a validated clinical multi-agent system.

## 2. Tests

* Layout: top-level unit and API tests, plus `tests/phase3`-`tests/phase6`. The 26 skips are live-service tests (22 RLS, 4 Supabase/Gemini), unchanged since Phase 4A. **No test depends on the real UCI files**, so a green pytest run is silent about real-data validity; the workflow's exit code carries that.
* Phase 6 tests: metrics against scikit-learn and hand calculation, candidate/protocol pinning, an end-to-end report pipeline on forged data (about 2.5 minutes), warning handling, sample-size arithmetic, experiment and claims registers, documentation numbers and no-claim scan.
* Weak spot: the end-to-end test uses forged data matched to stand-in artifacts, so it proves the code path and the fail-closed gates, not model behaviour.
* Contract change disclosed: provenance contract 1.1.0 is stricter (adds intended use, split identifiers, performance link). Two Phase 5 tests that built a "complete" record needed a validation-run link; they were updated to supply one and four tests were added. The contract was tightened, not an assertion weakened.

## 3. CI

| Finding | Action |
|---|---|
| No CI workflow existed (F12) | Added `.github/workflows/backend-ci.yml`: a `tests` job (Python 3.13, `requirements.lock.txt`, pytest) and a `real-data-validation` job that maps workflow exit codes to a step summary, emits a warning for `NOT_RUN`, fails on exit 2 only if `require_real_data` is set, and uploads the manifest |
| Never executed on GitHub (push 403, not authorised) | YAML parsed locally only; claim CL20 is `NOT_RUN` |
| Frontend tests (Vitest, Playwright against a mocked API) | Not part of this workflow; not audited in Phase 6 |

## 4. Statistical infrastructure

`ml/evalkit` provides grouped splits, cluster bootstrap and permutation tests; Phase 6 added `clinical_metrics.py` (12 metrics, paired differences,
decision curves, calibration, subgroups with minimum-size gates) and `sample_size.py`. Known limits: percentile bootstrap with 42 clusters can undercover; the paired
bootstrap resamples subjects, so the repeated seeds are not independent replications; decision-curve interpretation needs a clinical action; no multiplicity correction across the many comparisons (the voice-value rule is a single prespecified intersection).

## 5. Reproducibility

Strengths: frozen and hashed protocols, run manifests with commit and dirty flag, fixed seeds, pinned lock file, NOT_RUN records committed, registers derived from files.
Gaps: the shipped artifacts were pickled under scikit-learn 1.9.0 and load under 1.9.1 with a version warning (counted, not hidden); bit-identical reruns across platforms are untested.

## 6. Defects found and resolved in Phase 6

| ID | Finding | Resolution |
|---|---|---|
| F7 | Four validation modules suppressed all warnings globally at import time | Replaced by scoped, counted handling (`ml/evalkit/warnings_util.py`); subprocess test proves no global ignore filter remains |
| F8 | Phase 5 `REPRODUCIBILITY.md` said the registered workflow runs an unauthenticated file with a label; the code refuses it (exit 3). Only `phase4a` runs it, labelled | Corrected in `DATA_AUTHENTICATION.md`; the Phase 5 file is kept as history |
| F9 | README quoted "551 passing" tests | Corrected with the measured count (see `FINAL_READINESS_REPORT.md`) |
| F10 | Provenance sidecars lacked intended use, split identifiers and a link to a validation run | Contract 1.1.0; shipped artifacts stay `INCOMPLETE_DECLARED` with the reason recorded |
| F11 | No machine-readable list of experiments or claims | `ml/validation/experiment_registry.py` and `claims_register.py` with tests |

## 7. Findings left open

Consensus weights unvalidated (needs outcome data); consensus is single-voter in effect; real-data validation not run; CI not executed remotely; frontend not re-audited;
`backend/patient_body.json` is a tracked test fixture containing a synthetic "Test Patient Record" (no real person's data found); `backend/neuromas.db` is git-ignored.
