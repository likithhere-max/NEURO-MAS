# Integration and Test Report (Phase 2)

Date: 2026-10-23. Environment: Linux sandbox, Python 3.13.17, pytest 9.1.1, scikit-learn 1.9.1, Node 20, Vitest 2.1.9,
Chromium via Playwright for Python. Network: outbound HTTPS only through an allow-list proxy (package registries); UCI,
PubMed/WHO/ICD, Supabase and Google endpoints were not reachable.

## 1. Rule applied
A test that uses a stand-in for a dependency is described as testing **our code against that stand-in**. No test in this
report verifies a live Supabase project, a live Gemini model, a live SMTP server, a live Redis, or any real-patient
behaviour.

## 2. Actual results (commands and output)

```text
$ cd backend && python -m pytest tests -q --cov=app --cov=ml -p no:cacheprovider
551 passed, 26 skipped, 36 warnings in 92.40s (0:01:32)
TOTAL                                          5357    744    86%

$ cd frontend && npx tsc --noEmit            -> exit 0
$ npx vitest run                              -> Test Files 4 passed (4); Tests 24 passed (24)
$ npx next build                              -> exit 0
$ bash e2e/run.sh                             -> 8/8 checks passed   (mocked API)
$ python backend/scripts/scan_secrets.py      -> 7 finding(s) across 11 commit(s)   [all: shared test password literal, in history]
$ python -m ml.validation.validate_models artifacts   -> completed (manifest hashes match; sklearn 1.9.0→1.9.1 version warnings)
$ python -m ml.validation.validate_models fetch       -> FAILED: URLError: Tunnel connection failed: 403 Forbidden
```

The 26 skips are: 22 parametrised cases in `tests/rls_live/test_rls_matrix.py` (no separate Supabase test project) and 4
live Supabase/Gemini tests (`NEUROMAS_LIVE_TESTS=1` + credentials). **Skipped is not passed.**

Coverage is line coverage only (86 %); it says which lines ran, not that behaviour is right. Lowest: `ml/simulation/*` scripts
(33 % / 0 % — covered indirectly by one seed-reproducibility test), `app/services/llm_provider.py` 35 %,
`symptom_extraction.py` 35 %, `routes/patients.py` 36 %, `routes/chat.py` 37 %, `core/supabase_client.py` 38 %,
`ml/validation/data.py` 63 % (download path unexercised), `gemini_service.py` 68 %, `services/auth.py` 68 %.

## 3. What each group of tests establishes

| Area | Test files (count) | Real or stand-in | What a pass shows |
|---|---|---|---|
| Evaluation kit, leakage guards, runner | `test_ml_evalkit` (26), `test_ml_validation_runner` (9) | real code, synthetic/fixture data | splitters reject leakage; metrics return `None` for undefined cases; reference values match hand computation; runner labels synthetic output "NOT_EVIDENCE" |
| OOD / abstention | `test_ood` (18) | real code and the real shipped model scalers | foreign/invalid inputs abstain; training mean is in-domain |
| Longitudinal maths | `test_longitudinal_math` (30), `test_personalized_anomaly` (12), `test_longitudinal_integration` (7), `test_metric_semantics` (25), `test_metric_thresholds` (8), `test_math_properties` (8) | real code | formulas match hand examples and `scipy`; properties hold; provisional/approved thresholds behave as specified |
| Consensus / agents | `test_consensus` (10), `test_consensus_evidence` (20), `test_agents` (5), `test_agent_typed_statuses` (3) | real code | ranking rules, no double counting, non-voting statuses |
| Orchestration | `test_orchestrator` (9), `test_orchestrator_concurrency` (20) | real orchestrator, **fake agents** that sleep/hang/fail | ordering, concurrency bound, deadlines, cancellation, fail-closed emergency. No load test; real agents' latency not measured |
| Persistence / routes | `test_persistence_payload` (4), `test_workflow_routes` (25) | real routes, **in-memory fake Supabase client** | idempotency, partial-failure markers, emergency not swallowed, response contract. Says nothing about real RLS, SQL, or the RPC |
| Medication / KB | `test_drug_rules` (24), `test_ddi` (20), `test_kb_integrity` (17) | real code; link checker also run against a **real local HTTP server** (200/404/301/refused) | distinct DDI statuses, absence ≠ safe; link states classified correctly. The real KB URLs could not be checked |
| Security / privacy | `test_security` (20), `test_auth_jwt` (8), `test_privacy` (30), `test_scan_secrets` (11), `test_safe_load` (13), `test_ratelimit` (7), `test_api_contracts` (11), `test_integrations_hardening` (17) | real code; Supabase client **mocked**; Redis **fake** | signup/consent/rate-limit/redaction/loader/contract behaviour; scanner finds planted secrets |
| Notifications | `test_notification_dispatch` (35), `test_actions_notifier` (4) | **mock provider**; SMTP provider exercised against a `smtplib` stand-in | state machine, idempotency, bounded retry, escalation, PHI-free audit. **No message was sent anywhere** |
| RLS | `tests/rls_live/test_rls_matrix.py` (25 collected) | harness guards run; live cases **skipped** | the guards refuse unsafe targets. Tenant isolation is **not verified** |
| Frontend | Vitest (24), Playwright smoke (8 checks) | jsdom / Chromium with a **mocked API** | error-message mapping, login behaviour, delivery/consensus wording, abstention rendering; the old ML Lab page fails the smoke test and the fixed page passes |

## 4. Defects found and fixed during Phase 2 (each with a failing test first)
| ID | Root cause | Fix | Evidence |
|---|---|---|---|
| D-1 | `scan_secrets.py` passed PCRE-only syntax to `git grep -E`; git errored and the script printed "0 findings" for any input | rewrote to scan blobs in Python, exit 2 on failure, added password-literal pattern | `test_scan_secrets.py` plants secrets (11 cases); real run now reports the exposed test password. **The earlier "no secrets found" statement was wrong and is withdrawn.** |
| D-2 | `vitals`/`wearable` were `dict[str, Any]`: NaN/Infinity accepted (NaN > x is False ⇒ reads as "normal"), unbounded size/depth | validator `_bounded_finite` | 6 parametrised rejection cases + accept-case |
| D-3 | ML Lab page called `null.replace` / printed "predicted: null" when a model abstained | `AbstainNotice` component and guards | Playwright smoke fails on the pre-fix page, passes on the fixed one |
| D-4 | Agent panel rendered `timed_out`/`abstained`/`insufficient_data` as neutral badges | shared `agentStatusStyle` + backend↔frontend vocabulary contract test | `test_api_contracts.py`, `labels.test.ts` |
| D-5 | Default config (`NOTIFICATION_CHANNEL=none`) still requested admin credentials and logged an ERROR on every request | admin client created only when a channel is configured | `test_default_config_does_not_log_an_error_or_need_admin_credentials` |
| D-6 | Agents returned `completed` with confidence 0 when they could not assess anything | `insufficient_data` / `abstained` | `test_agent_typed_statuses.py` |
| D-7 | Uniform up-scaling of reliability weights was assumed scale-free; weights are clipped to [0, 1] | property corrected and the clip documented | `test_math_properties.py` |
| D-8 | Earlier in Phase 2 (details in commit messages): redaction order bug (names masked inside e-mail before e-mail pattern), incomplete fake client hiding a persistence-marker path, hand-calculation slip in a consensus test (code was right) | fixed | respective tests |

## 5. What was NOT tested, and why
* Any behaviour against real Supabase (RLS, SQL migrations, RPC, e-mail confirmation setting): no project/credentials.
* Live Gemini behaviour (prompt-injection resistance, answer quality, consent-gate end-to-end with a real client).
* Real SMTP delivery, Redis, multi-process rate-limit sharing, clock skew.
* Load, soak, and latency of real agents; behaviour under many concurrent users.
* Real-data ML performance; OOD detector error rates on real data.
* Emergency-detector recall on real patient language.
* Accessibility audit beyond `role="alert"/"status"` checks; cross-browser testing (Chromium only).
* Security review / penetration test; dependency vulnerability scan.
