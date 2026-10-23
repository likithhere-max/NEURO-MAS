# Run and Reproduction (Phase 2)

All commands are from the repository root unless stated. Python 3.13 and Node 20+ were used.

## 1. Install
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt            # exact versions the suite passed against
# or: pip install -r requirements.txt -r requirements-dev.txt
pip install -r requirements-optional.txt        # only if you will set RATE_LIMIT_REDIS_URL
cd ../frontend && npm ci
```

## 2. Run the system (development)
```bash
cd backend
# create backend/.env (never commit it) with at least: SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, SUPABASE_SECRET_KEY, GEMINI_API_KEY
# (names from app/core/config.py; there is no .env.example in the repo)
ENVIRONMENT=development uvicorn app.main:app --reload
cd ../frontend && cp .env.local.example .env.local && npm run dev   # http://localhost:3000
```
New accounts need e-mail confirmation. For a local demo only: `AUTO_CONFIRM_EMAIL=true` (honoured only with `ENVIRONMENT=development`).

## 3. Tests (what was run for this report)
```bash
cd backend
python -m pytest tests -q --cov=app --cov=ml          # 551 passed, 26 skipped, 86 % line coverage (see INTEGRATION_TEST_REPORT.md)
cd ../frontend
npx tsc --noEmit
npx vitest run                                        # 24 passed
npm run build && bash e2e/run.sh                      # mocked-API browser smoke, 8/8
```
The 26 skips: 22 × `tests/rls_live` (no separate Supabase test project) and 4 × live Supabase/Gemini tests
(`NEUROMAS_LIVE_TESTS=1` plus configured credentials needed).

## 4. Behaviour changes versus Phase 1
See SECURITY_AND_PRIVACY.md §4. In addition: consensus values are an uncalibrated support index (no `%`), agents can
report `abstained` / `insufficient_data` / `timed_out`, actions can be `queued` / `sent_unconfirmed` / `delivery_failed`,
and the ML demo can return `abstained: true` with null label/probabilities.

## 5. Environment variables added in Phase 2
| Variable | Default | Purpose |
|---|---|---|
| `WORKFLOW_DEADLINE_SECONDS`, `AGENT_TIMEOUT_SECONDS`, `MAX_AGENT_CONCURRENCY` | see `app/core/config.py` | orchestration bounds |
| `PERSIST_VIA_RPC` | false | use the atomic `persist_workflow_atomic` SQL function (apply migration first) |
| `DDI_SNAPSHOT_PATH` | empty | path to an owner-supplied versioned DDI snapshot JSON |
| `EXTERNAL_PROCESSING_MODE` | `redacted` | `disabled | redacted | unredacted` |
| `REQUIRE_PATIENT_CONSENT` | false (was true until 2026-10-24; owner decision: AI always on) | when true, gate patient-scoped LLM calls on `patients.external_ai_consent` |
| `AUTO_CONFIRM_EMAIL` | false | development only |
| `NOTIFICATION_CHANNEL` (`none | smtp`), `SMTP_*`, `NOTIFICATION_MAX_ATTEMPTS`, `NOTIFICATION_ATTEMPT_TIMEOUT_S`, `NOTIFICATION_BACKOFF_BASE_S` | `none` | notification delivery |
| `METRIC_STABLE_BAND`, `METRIC_ADVERSE_THRESHOLD`, `METRIC_MCID_PATH` | 0.05, 0.20, empty | provisional change thresholds; MCID file entries need `approved_by` and `source` |
| `WORKFLOW_RATE_LIMIT_PER_MINUTE`, `CHAT_RATE_LIMIT_PER_MINUTE`, `AUTH_RATE_LIMIT_PER_MINUTE` | 6, 20, 10 | per-user / per-IP limits (0 disables) |
| `TRUSTED_PROXY_COUNT`, `RATE_LIMIT_REDIS_URL` | 0, empty | proxy-aware client IP; shared limiter store |
| `NEUROMAS_TEST_PROJECT=yes`, `NEUROMAS_TEST_PASSWORD`, `NEUROMAS_RLS_TEST_*` | unset | live scripts / RLS matrix (test project only) |

## 6. Reproducing the numerical evidence
```bash
cd backend
PYTHONPATH=. python -m ml.simulation.anomaly_study --reps 400 --seed 20261023 --out ml/simulation/results/anomaly_study.json
PYTHONPATH=. python -m ml.simulation.point_test_repeated_review --reps 1000
PYTHONPATH=. python -m ml.validation.validate_models artifacts
python scripts/scan_secrets.py                      # commit first; exit 1 = findings, 2 = scan could not run
python scripts/check_kb_links.py                    # needs internet; unreachable != dead
```
Real-data ML validation: see ML_EXPERIMENT_PROTOCOL.md §10 (needs internet; **not run here**).

## 7. Applying SQL (do on a TEST project first)
`scripts/migration_phase2_persistence.sql`, `migration_phase2_privacy.sql`, `migration_phase2_notifications.sql`.
They were written without database access and **never executed**; read them, verify column types against your live
schema (especially `workflows`, `actions`, `patients` ids), then apply.
