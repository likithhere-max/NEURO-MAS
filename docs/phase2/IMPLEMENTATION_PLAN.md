# NEURO-MAS Phase 2 — Implementation Plan and Status

*Last updated 2026-10-23. Branch `phase2/advanced-ml-hardening` (local commits only; nothing has been pushed — see
OUTSTANDING_ISSUES.md).*

## 0. Scope statement

Phase 2 continues the Phase 1 audit repo (no rebuild). The goal is **demonstrated correctness within a clearly stated
scope**. It is not a claim of clinical validity, regulatory readiness, security certification or production readiness,
and nothing here should be quoted as one.

Status words used in every Phase 2 document:

| Word | Meaning |
|---|---|
| **COMPLETED** | Implemented and exercised by automated tests that actually ran in this environment. |
| **COMPLETED (mock/in-process)** | As above, but the dependency (database, SMTP, Redis, browser API) was replaced by a stand-in. Says nothing about the live service. |
| **WRITTEN, NOT EXECUTED** | Code/SQL exists but could not be run here (no credentials/network/test project). |
| **BLOCKED** | Needs something only the owner can supply (data, credentials, decision). |
| **NOT RUN / FAILED** | A procedure that was attempted and did not produce a result (the failure record is kept). |
| **CLINICALLY UNVALIDATED** | Applies to every model, threshold and score in the system, without exception. |

## 1. Baseline (Phase A)

Measured at the start of Phase 2 (end of Phase 1 audit, commit `1be5e52`): Phase 1 suite passing; real-data ML
validation never run; consensus produced a percentage from one voting agent; orchestrator strictly serial; persistence
as several non-transactional inserts; signup auto-confirmed e-mail; no notification channel; PHI sent to the LLM
unconditionally; rate limit per-process keyed on socket peer.

Environment facts that bound what can be claimed: the sandbox has outbound access only through an allow-list proxy.
UCI, GitHub (read for datasets), PubMed/ICD/KB URLs and any Supabase project were **unreachable** (HTTP 403 from the
proxy), so nothing that needs those could be executed (details in INTEGRATION_TEST_REPORT.md).

## 2. Work packages and where each ended up

| Phase | Deliverable | Status | Evidence (commit / file) |
|---|---|---|---|
| B1–B3 | Leakage-safe splitters, nested group CV, metrics that return *undefined* instead of fabricating, cluster bootstrap, permutation test, prevalence-adjusted PPV/NPV | COMPLETED | `ml/evalkit/*`, `tests/test_ml_evalkit.py`, `9ae3460` |
| B4 | Validation runner for the two shipped models | COMPLETED (code) / **real-data run: BLOCKED** | `ml/validation/*`; `results/fetch_FAILED.json` records the failed download |
| B5 | Artifact integrity run on the repo's own files | COMPLETED (real run) | `ml/validation/results/artifacts.json` |
| B6 | OOD / abstention for the two demo models | COMPLETED | `app/ml/ood.py`, `tests/test_ood.py` |
| C1 | Longitudinal methods chosen by simulation | COMPLETED (simulation only) | `app/ml/longitudinal.py`, `ml/simulation/results/anomaly_study.json`, MATHEMATICAL_VALIDATION.md |
| C2 | Configurable provisional thresholds; MCID only if clinician-approved | COMPLETED | `app/core/metric_semantics.py`, `tests/test_metric_thresholds.py` |
| C3 | Property-based math regression tests | COMPLETED | `tests/test_math_properties.py` |
| D1–D3 | Typed agent statuses, hypothesis-specific typed evidence, no consensus probability | COMPLETED | `app/core/evidence.py`, `app/core/consensus.py`, `tests/test_consensus_evidence.py` |
| D4 | DAG orchestrator, bounded concurrency, deadlines, cancellation, fail-closed emergency | COMPLETED | `app/orchestrator/orchestrator.py`, `tests/test_orchestrator_concurrency.py` |
| D5 | Idempotent, atomic persistence | COMPLETED (in-process fake) / **RPC SQL WRITTEN, NOT EXECUTED** | `scripts/migration_phase2_persistence.sql`, `tests/test_persistence_payload.py` |
| E1 | DDI source interface, honest statuses | COMPLETED (interface) / **no DDI source shipped** | `app/services/ddi.py`, `tests/test_ddi.py` |
| E2 | KB link-liveness checker + review queue | COMPLETED (tool) / **live check could not run** | `scripts/check_kb_links.py`, `data/kb_review_queue.json` |
| F1 | RLS test matrix for a *separate test project* | WRITTEN; 3 guard tests run, **22 live tests skipped** | `tests/rls_live/test_rls_matrix.py` |
| F2 | Remove unconditional e-mail-confirm bypass | COMPLETED | `app/api/routes/auth.py`, `tests/test_security.py` |
| F3 | Secret scan; hard-coded credential cleanup | COMPLETED (scan) — **revocation is an owner action** | `scripts/scan_secrets.py`, SECURITY_AND_PRIVACY.md |
| F4 | Consent gate, redaction, external-processing policy | COMPLETED (code) / **privacy SQL WRITTEN, NOT EXECUTED** | `app/core/privacy.py`, `tests/test_privacy.py` |
| G | Notification outbox / provider abstraction | COMPLETED (mock + smtplib stand-in) / **no live delivery** | `app/services/notification_*.py`, `tests/test_notification_dispatch.py` |
| H1 | Shared rate limiting + trusted proxy + per-user limits | COMPLETED (Redis via fake) | `app/core/ratelimit.py`, `tests/test_ratelimit.py` |
| H2 | Safer model loading | COMPLETED (defence in depth, not a proof) | `app/ml/safe_load.py`, `tests/test_safe_load.py` |
| H3 | Dependency pinning | COMPLETED (not hash-pinned) | `requirements.lock.txt` |
| H4 | API contract tests | COMPLETED | `tests/test_api_contracts.py` |
| H5 | Frontend unit tests + mocked-API browser smoke | COMPLETED (mocked API) | `frontend/tests/*`, `frontend/e2e/*` |
| I | Final regression with coverage | COMPLETED | INTEGRATION_TEST_REPORT.md |
| J | Documents | COMPLETED (this folder) | — |

## 3. Execution order actually followed

Baseline → ML evaluation kit + leakage guards → mathematics → OOD → longitudinal/anomaly → typed evidence and consensus →
idempotency/concurrency/persistence → Supabase authorization (tests + SQL) → medication/KB integrity → consent/privacy →
notification provider → frontend contracts and E2E → full regression and documents.

## 4. Per-issue reporting convention

Each material change is listed with: ID, root cause, files, implementation, justification, tests, actual result,
limitations, owner action. The full list is in `OUTSTANDING_ISSUES.md` (open items) and in the mapping table at the end of
that file (Phase 1 O1–O24 → Phase 2 outcome). Behaviour changes that the owner must know about are in
RUN_AND_REPRODUCTION.md §4.

## 5. Constraints honoured

No push, force-push, history rewrite or reset of existing work; no production deployment; no purchase; no real patient
data transmitted; no real person contacted; no live notification sent; no destructive test against any production
system; no secret printed in any log or report. The one external request made (UCI download) was refused by the
proxy and is recorded as FAILED.
