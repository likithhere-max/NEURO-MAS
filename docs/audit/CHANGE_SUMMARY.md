# NEURO-MAS — Change Summary (Deliverable 10)

> **Point-in-time record (Phase 1 audit, 2026-10-23, commit `1be5e52`).** Parts of this document have been superseded by Phase 2 — see `docs/phase2/IMPLEMENTATION_PLAN.md` and `docs/phase2/OUTSTANDING_ISSUES.md`. Where the two disagree, **the Phase 2 document is current**. Inline notes marked **[Phase 2]** flag changed items.

All changes are **uncommitted** in the working tree (`50 files changed, +1304/−713` for tracked files plus 19 untracked paths). Nothing has been pushed.

## Public-API / schema / dependency changes (flagged, as requested)
| Change | Compatibility |
|---|---|
| `POST /auth/signup` accepts optional `invite_code`; clinician signup refused unless code matches or `ENVIRONMENT=development` | **Behaviour change** for clinician registration |
| `POST /auth/*` rate-limited (HTTP 429) | additive |
| Chat response adds `is_emergency`; chat message max 2000 chars | additive / tightening |
| Workflow response adds `persisted`, `degraded`, `failed_agents`, `emergency_screen`; result_json adds `margin` etc. | additive |
| Request validation tightened (ranges, finite numbers, lengths); 422 body reduced to loc/msg/type | tightening |
| Actions: status `executed` only when delivery confirmed, else `proposed` | **semantic change** (the honest one) |
| Approve/acknowledge require clinician; approve only `pending_approval` | tightening |
| **No DB schema change.** `workflows.status` stays `completed`; degraded info lives in `result_json` | — |
| Dependencies: + `PyJWT[crypto]`, scikit-learn pinned `>=1.9.0,<1.10`; − python-jose, passlib, bcrypt, aiosqlite, greenlet, sqlalchemy, websockets, python-multipart; pytest moved to `requirements-dev.txt`. Config: − `database_url`; + `agent_timeout_seconds`, `max_free_text_chars`, `max_chat_message_chars`, `clinician_invite_code`, `auth_rate_limit_per_minute` | verify your deployment's install step uses the new files |

## New files
`core/metric_semantics.py`, `core/ratelimit.py`, `services/notifier.py`, `ml/models/MANIFEST.json`, `ml/validation/validate_models.py`, `requirements-dev.txt`, 12 new test modules, and `docs/audit/*`.

## Modified (by area)
* **Safety:** `clinical_rules.py` (emergency regex families, negation, drug normalisation), `orchestrator.py` (pre-scan, timeout, fail-closed, degraded), `actions.py`, `agents/emergency.py`, `chat_assistant.py`.
* **Maths:** `consensus.py`, `personalized_anomaly.py`, `metric_semantics.py` (+ `longitudinal.py`, `personalization.py`, `repository.py` using it).
* **Agents:** `data_intake`, `signal_analysis`, `risk`, `medication`, `knowledge`, `diagnostic` (honest confidence, validation, coverage).
* **Security:** `auth.py` (routes + service), `alerts.py`, `patients.py`, `workflows.py`, `main.py`, `supabase_client.py`, `ml_demo.py`, `gemini_service.py`, schemas.
* **Frontend:** `api.ts`, `login/page.tsx`, `ActionsPanel`, `AgentActivityPanel`, `ConsensusPanel`, `PatientContext`, patient page, evidence page.
* **Docs:** README, `docs/SIMULATED_VS_REAL.md`, `docs/ARCHITECTURE.md` §7 updated to match reality.

## Deleted
`backend/app/db/`, `backend/app/seed/` (dead SQLAlchemy scaffolding; `create_all` also removed from `main.py`), `tests/test_neuromas.db`.

## Not changed on purpose
Supabase schema/RLS, `KNOWN_INTERACTIONS` content (no unsourced clinical content added), the ML model files, the KB data.

## How to adopt
1. Review `git diff`; run the test suite (RUN_AND_REPRODUCTION §3).
2. Set `ENVIRONMENT`, `CLINICIAN_INVITE_CODE`, `CORS_ORIGINS` on the deployed backend.
3. Commit (suggested: split into safety / maths / agents / security / frontend / docs). I have not committed; pushing needs your go-ahead and write access.
