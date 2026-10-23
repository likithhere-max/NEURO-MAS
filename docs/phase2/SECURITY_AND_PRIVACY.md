# Security and Privacy (Phase 2, F/G/H)

**This document describes engineering controls and their test status. It makes no compliance claim** (not HIPAA, GDPR,
India's DPDP Act, ISO 27001, SOC 2, or a medical-device regulation) **and no claim that the system is secure.** The system has
not had an independent security review or penetration test. Do not use it with real patient data until the "Owner actions"
below are done and reviewed by qualified people.

## 1. Threat model used
Assets: patient records and observations (special-category health data), clinician actions, credentials, model artifacts.
Actors: anonymous internet users; authenticated patient/caregiver; authenticated clinician (assigned to other patients); a
compromised dependency/artifact; an operator who mis-configures. Out of scope here: host/OS hardening, network
architecture, Supabase platform security, key management processes, physical security, insider threats beyond RLS.

## 2. Controls and their status

| ID | Control | Where | Status |
|---|---|---|---|
| S1 | Row-Level-Security authorization matrix (clinician↔own patients only; no cross-patient child-table access or insert; unrelated users see nothing; anonymous sees nothing; no self-escalation of role; no cross-profile read; consent flag not writable by unrelated users) | `tests/rls_live/test_rls_matrix.py` | **WRITTEN; NOT EXECUTED.** 3 harness-guard tests run; the 9 live test functions (22 parametrised cases) are skipped because no separate test project exists. The harness refuses to run unless a test project is configured, its URL differs from the app's own `SUPABASE_URL`, `NEUROMAS_TEST_PROJECT=yes` is set and a strong password is supplied. A pass would show only that those assertions hold on that project's policies. |
| S2 | The RLS policies and schema themselves are **not in this repository** | — | **OPEN.** `supabase db dump --schema public > supabase/schema.sql` and review them. Nothing in Phase 2 can establish tenant isolation without that. |
| S3 | Signup no longer auto-confirms e-mail | `app/api/routes/auth.py` | **COMPLETED** (mock-tested). Production path returns `202 {"status":"confirmation_required"}` and issues no token; `AUTO_CONFIRM_EMAIL=true` works only when `ENVIRONMENT=development` and the server refuses to start otherwise. **Requires** Supabase "Confirm email" to be ON in the project (unverifiable here). |
| S4 | MFA | — | **EVALUATED, NOT IMPLEMENTED.** Supabase Auth offers TOTP MFA; the API verifies the access token's JWT but does not currently check the `aal` (assurance level) claim. Recommended minimum: require `aal2` for clinician actions that approve restricted actions and for clinician signup/login, and enrol clinicians via the Supabase MFA API. This is a product/UX decision with account-recovery implications, so it was not done unilaterally. |
| S5 | Secret scanning of git history | `scripts/scan_secrets.py`, `tests/test_scan_secrets.py` | **COMPLETED, with a correction.** An earlier version silently found nothing (invalid `git grep` regex); it was rewritten and now has positive-control tests. Real run on this repo: **7 findings, all the same shared test-account password literal in 7 scripts, present in history since 2026-09-09** (the literal was removed from the working tree on 2026-10-23, the history was **not** rewritten). Key-shaped secrets (JWT, `sb_secret_`, Google key, GitHub token, `sk-`, DB URL with password, private key): none found — within the limits of those patterns. |
| S6 | Hard-coded credentials in live scripts | `scripts/_testcfg.py`, 7 scripts | **COMPLETED.** Scripts read `NEUROMAS_TEST_PASSWORD` and require `NEUROMAS_TEST_PROJECT=yes`; no password literal remains in the tree. |
| S7 | Consent gate for external (LLM) processing | `app/core/privacy.py`, `app/ai/gemini_service.py` | **COMPLETED** (30 tests). Mode `disabled | redacted (default) | unredacted`; patient-scoped calls require `patients.external_ai_consent = true` (default false ⇒ fail closed); a call with **no** declared scope is blocked; non-patient work must declare `system_scope`. The symptom-assessment route returns 403 without consent. Audit events hold counts only, never text. |
| S8 | Redaction | same | **Best-effort** masking of e-mails, URLs, SSN-shaped numbers, phone numbers, dates, long numeric IDs and the patient's own known name parts. It will miss free-text names, addresses, employers. **It is not de-identification.** |
| S9 | Consent column / API | `scripts/migration_phase2_privacy.sql`, `PUT /patients/{id}/ai-consent` | **SQL WRITTEN, NOT EXECUTED.** Until it is applied, every patient reads as "no consent" and external narration is blocked (safe default, but a visible behaviour change). The existing RLS on `patients` must be reviewed so only the patient or treating clinician can change the flag. |
| S10 | Rate limiting | `app/core/ratelimit.py` | **COMPLETED (Redis through a fake).** Auth endpoints per client IP; workflow/symptom/chat per authenticated user; `Retry-After` on 429. Behind proxies, set `TRUSTED_PROXY_COUNT=N` (rightmost-Nth `X-Forwarded-For` hop is trusted; with 0 the header is ignored). If Redis is configured but fails, the limiter falls back to the per-process store and logs a warning (fail-open by design). Not a substitute for an edge/WAF limiter. |
| S11 | Input bounds | `app/schemas/api.py` | **COMPLETED.** Free-form `vitals`/`wearable` dicts reject NaN/±Infinity (which would silently pass every `x > threshold` check), >100 keys, >3 levels, strings >500 chars, lists >1000, non-JSON types. |
| S12 | Model artifact loading | `app/api/routes/ml_demo.py`, `app/ml/safe_load.py` | **COMPLETED (defence in depth).** SHA-256 against `ml/models/MANIFEST.json`, then a restricted unpickler that resolves only numpy / scikit-learn / `scipy.sparse` / joblib array types and refuses `os`, `subprocess`, `eval`, `numpy.load`, `sklearn.datasets`, test helpers (8 policy cases + 2 malicious-pickle tests). **Not a proof of safety**: pickle is still pickle, and a malicious file whose hash also appears in a tampered manifest would be limited, not stopped, by the allow-list. |
| S13 | Dependency pinning | `requirements.lock.txt` | **COMPLETED, not hash-pinned.** Records the versions the suite passed against. No vulnerability scan (`pip-audit` etc.) was run — the sandbox has no vulnerability database access. |
| S14 | CORS | `app/core/config.py` | **OPEN.** The default list still contains `"*"` (a 2026-09-02 convenience). Bearer-token APIs are less exposed than cookie APIs, but pin `CORS_ORIGINS` for production. |
| S15 | Frontend token handling | `frontend/lib/api.ts` | Token is held in memory and `sessionStorage` (cleared when the tab closes). It is readable by any script on the origin, so an XSS bug would expose it. An HttpOnly-cookie session with CSRF protection would be stronger; not done. |
| S16 | Logs | various | Audit and log events were designed to carry ids, states and error classes only. No automated test proves that **no** PHI can reach any log line anywhere; `notification_*` and `privacy_*` paths are tested. |

## 3. Notification security (G)
See `app/services/notification_dispatch.py`. Messages are generic (no PHI); recipients are opaque user ids resolved to an
address only inside the provider; the outbox table is server-owned (RLS enabled with **no** policies ⇒ end users and
anonymous callers can neither read nor write it); idempotency key `action_id:channel` prevents duplicate sends; only
transient failures are retried (bounded); a permanent failure is escalated through a critical alert; "executed" is shown
only for `delivered`/`acknowledged`. The SMTP provider is **real code verified only against a stand-in for `smtplib`**; no
message has ever been sent from this repository. `NOTIFICATION_CHANNEL` defaults to `none`.

## 4. Behaviour changes the owner must know about
1. New accounts require e-mail confirmation (unless `AUTO_CONFIRM_EMAIL=true` in development).
2. External LLM use is blocked until each patient has `external_ai_consent = true` (needs the privacy migration) or the owner sets `REQUIRE_PATIENT_CONSENT=false` / `EXTERNAL_PROCESSING_MODE=disabled`. Symptom assessment returns 403 without consent.
3. Live scripts refuse to run without `NEUROMAS_TEST_PROJECT=yes` and `NEUROMAS_TEST_PASSWORD`.
4. Per-user rate limits apply to workflow (6/min), symptom (shared with workflow) and chat (20/min) endpoints.
5. Oversized or non-finite observation payloads now return 422.

## 5. Owner actions (cannot be done from here)
1. **Treat the shared test password as exposed.** It has been public since 2026-09-09. Disable or rotate every account that used it (`apitest@example.com`, `neuromas.test.user@example.com`, `neuromas.test.user2@example.com`) and any account elsewhere sharing it. Removing the string from files or rewriting history does **not** revoke anything.
2. Decide whether to rewrite history to remove the literal. That needs your explicit decision (it invalidates every clone and fork); it was not done.
3. Apply the three SQL migrations to a **test** Supabase project first, run `tests/rls_live`, review results, then production. Export the real schema/policies into the repo.
4. Turn on "Confirm email" in Supabase; decide MFA policy (S4).
5. Set `ENVIRONMENT=production`, `CLINICIAN_INVITE_CODE`, pinned `CORS_ORIGINS`, `TRUSTED_PROXY_COUNT`, optionally `RATE_LIMIT_REDIS_URL`; keep keys in the environment only.
6. Confirm the LLM provider's data-processing terms for your key/plan and your legal basis for sending health data; the consent gate is a technical control, not that analysis.
7. Rotate any key that was ever committed or pasted anywhere, regardless of this scan's result.
8. Run a dependency vulnerability scan and an independent security review before any real-patient use.
