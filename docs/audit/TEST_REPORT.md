# NEURO-MAS — Automated Test Report (Deliverable 6)

> **Point-in-time record (Phase 1 audit, 2026-10-23, commit `1be5e52`).** Parts of this document have been superseded by Phase 2 — see `docs/phase2/INTEGRATION_TEST_REPORT.md` and `docs/phase2/OUTSTANDING_ISSUES.md`. Where the two disagree, **the Phase 2 document is current**. Inline notes marked **[Phase 2]** flag changed items.

All numbers below were produced by commands run in this session (2026-10-23, Python 3.13 venv,
Linux sandbox, no network access to Supabase/Gemini).

## 1. Results
```
cd backend && python -m pytest tests -q --cov=app
226 passed, 4 skipped, 6 warnings in 14.60s      TOTAL line coverage 81 %
```
* 4 skipped = `tests/test_api.py` (needs `NEUROMAS_LIVE_TESTS=1` and a configured Supabase). **They have not been run.**
* Lint: `ruff check --select F,E9` clean.
* Frontend: `npx tsc --noEmit` clean; `npx next build` succeeds. There are **no frontend unit/E2E tests** and no browser run was performed against a live backend.

## 2. Test inventory (230 collected)
| File | Tests | Covers |
|---|---|---|
| test_emergency_rules | 31 | stroke/seizure/breathing/GBS phrasings, curly apostrophes, negation, benign non-matches |
| test_metric_semantics | 25 | polarity registry, boundary 0.20 (IEEE-754), baseline 0, invalid |
| test_drug_rules | 24 | brand→generic, class expansion, allergy tokens, unassessed drugs |
| test_input_validation | 21 | NaN/inf/range/type errors, adherence units, de-dup |
| test_integrations_hardening | 17 | Gemini retry classification, timeout, manifest/hash |
| test_security | 14 | signup/invite, roles, rate limit, CORS, orphan cleanup |
| test_ml_demo_and_patients | 13 | model integrity check, clinician resolution |
| test_personalized_anomaly | 12 | t-threshold, small n, zero variance, non-finite |
| test_safety / test_consensus / test_orchestrator / test_workflow_routes / test_chat_assistant / test_auth_jwt / test_agents / test_clinical_rules / test_actions_notifier | 10 / 10 / 9 / 9 / 9 / 8 / 5 / 5 / 4 | policy engine, consensus, fail-closed, degraded, persistence fallback, chat guardrails, JWT claims, delivery honesty |
| test_api | 4 (skipped) | live end-to-end |

## 3. Technique and its limits
* An autouse fixture makes `_call_gemini` / `_call_embed` raise, so tests exercise the deterministic and fallback paths. **Consequence: no test exercises a real LLM or embeddings.**
* Supabase is replaced by fakes/monkeypatches. **Consequence: no test validates SQL/RLS/schema behaviour.**
* Many assertions are on data I authored (emergency phrases, drug names). They prove the code does what I intended, not that the intent covers real patient language.

## 4. Mutation sanity check
Running the new test files against the **original** `backend/app` (via `git stash`-style checkout of HEAD) gave **56 failing tests and 3 files that cannot import** (they reference functions that did not exist). So the suite detects the audited defects. This is not a full mutation-testing score.

## 5. Statistical checks
* Anomaly detector, Gaussian null, seeded Monte-Carlo (exploratory script run in this session, rates quoted as measured): false-positive rate 0.0518 / 0.051 / 0.0498 / 0.0496 for n = 3 / 5 / 10 / 30; power 0.92 for a 4-SD shift at n = 10.
* The suite itself contains `test_false_positive_rate_is_nominal_on_pure_noise` (parametrised over n) and `test_detects_a_real_shift_with_reasonable_power`, plus hand-calculation tests (`test_t_critical_matches_standard_table`, `test_effective_threshold_hand_calculation`, `test_hand_calculated_z_score`). Their tolerance bands are looser than the exploratory figures above.

## 6. Coverage gaps (lowest first)
`llm_provider` 35 %, `symptom_extraction` 35 %, `chat` route 36 %, `supabase_client` 38 %, `patients` route 32 %, `knowledge` route 55 %, `workflows` route 55 %, `repository` 58 %. These are the modules that need a live Supabase/Gemini.
