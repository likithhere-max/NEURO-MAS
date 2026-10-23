# NEURO-MAS — Security & Privacy Report (Deliverable 5)

> **Point-in-time record (Phase 1 audit, 2026-10-23, commit `1be5e52`).** Parts of this document have been superseded by Phase 2 — see `docs/phase2/SECURITY_AND_PRIVACY.md` and `docs/phase2/OUTSTANDING_ISSUES.md`. Where the two disagree, **the Phase 2 document is current**. Inline notes marked **[Phase 2]** flag changed items.

Scope: backend code in the repo. **Out of scope / not testable here:** the live Supabase project
(schema, RLS, auth settings), live Gemini, hosting (Netlify/ngrok). No penetration test was performed;
findings are from code review plus unit/integration tests with Supabase stubbed.

## 1. Authentication & authorisation
| Control | State | Evidence |
|---|---|---|
| JWT verification (Supabase ES256 via JWKS) | Requires `exp` and `sub`; issuer checked when `SUPABASE_URL` is set. **Not tested against the real JWKS endpoint.** | `test_auth_jwt.py` (8, self-signed key) |
| Clinician self-registration | Closed unless invite code matches (constant-time) or `ENVIRONMENT=development`. | `test_security.py` |
| Role checks on alert acknowledge / action approve | Clinician-only; approve only for `pending_approval` rows. | `test_security.py` |
| Patient→clinician assignment | Clinician only to self; others must name an existing clinician (admin-client lookup). | `test_ml_demo_and_patients.py` |
| Per-patient data access (patient A reading patient B) | Delegated to Supabase RLS via a per-request authed client. **RLS is not in the repo → UNVERIFIED (C4).** | — |
| Email verification | **Absent** (`email_confirm=True` on signup). Any string can be a "verified" email. | OPEN — **[Phase 2]** unconditional bypass removed; Supabase setting unverified |
| MFA / password reset / session revocation | Not implemented in the app (Supabase can provide). | OPEN — **[Phase 2]** MFA evaluated, not implemented (S4) |
| Rate limiting | Auth endpoints only; in-memory, per process; keyed on `request.client.host` (behind a proxy this may be the proxy IP). | `test_security.py`. PARTIAL |
| Service-role (admin) client | Used for registration and clinician lookup. Anything using it bypasses RLS — keep its use minimal; reviewed call sites only. | code review |

## 2. Input handling
* Pydantic bounds on every free-text field (5 000 chars), chat (2 000), ages, adherence, `top_k`; NaN/inf rejected globally.
* Validation errors return only `loc/msg/type` (no echo of submitted values → avoids reflecting PHI/NaN in logs/responses).
* SQL: all access is through the Supabase REST client (parameterised); no raw SQL in app code. Dead SQLAlchemy layer removed.
* ML artefacts: SHA-256 verified before unpickling (see ML report).
* Prompt injection (chat, symptom extraction): input delimited as untrusted data; output regex guardrail; deterministic emergency path never depends on the model. **Real-model resistance not tested.**

## 3. Secrets
* Scan of all tracked files: no API keys, JWTs or private keys found. `.env` was never committed (only `frontend/.env.local.example`).
* the shared test-account password (literal omitted; it remains in git history) is hard-coded in 7 files under `backend/scripts/` — in a **public** repo. If the account `apitest@example.com` (and `neuromas.test.user*@example.com`) exists in your production Supabase, **disable or rotate it**.
* `scripts/test_rls_deny.py` embeds a user UUID and runs against whatever Supabase the env points to (production).
* `backend/patient_body.json` is tracked; contents are synthetic.
* Git history was not rewritten; if you ever committed a secret earlier, rotate it (history not scanned beyond the current tree and what the push contained).

## 4. Transport / CORS
CORS default includes `"*"` (kept for the Netlify/ngrok demo). `allow_credentials` is now enabled only with explicit origins; a startup warning is logged in non-development. Set `CORS_ORIGINS` to the real frontend origin(s) before any real use.

## 5. Privacy / regulatory
| Issue | Detail | Status |
|---|---|---|
| **Patient text sent to Google Gemini** | Chat messages, symptom text, and embedding queries leave the system. No redaction, no consent screen. If a free-tier key is used, Google's terms may allow human review / training use of the data — **I did not verify the terms for your account**; check before any real user data. | OPEN (H20) | **[Phase 2]** consent gate + best-effort redaction + mode switch added (S7–S9); not a compliance control.
| DPDP Act 2023 (India) / GDPR / HIPAA | No claim of compliance is made or justified. Missing: consent capture, purpose limitation notice, data-subject access/erasure endpoints, retention policy, breach process, data-processing agreement with the LLM vendor, data residency decision. | OPEN |
| Audit trail | `audit_events` / `agent_event_log` written per workflow; integrity (tamper-evidence) not provided. | PARTIAL |
| Logging | Backend exception text is no longer returned to clients on auth routes; server logs may still contain PHI in error paths — not exhaustively reviewed. | PARTIAL |
| Data minimisation | Full free text stored and forwarded. | OPEN |
| Token storage (frontend) | Bearer token in browser storage (XSS-exfiltratable); no httpOnly cookie. | OPEN | **[Phase 2]** unchanged (S15).

## 6. Residual risk summary
1. Cross-patient isolation is only as good as RLS I could not see (**highest uncertainty**).
2. No real notification channel: emergencies are *detected and displayed* but not pushed to anyone (honestly labelled).
3. Third-party LLM data flow without consent.
4. Demo credentials in a public repo.
5. Open signup with unverified email for patient accounts.

## 7. Required deployment settings
`ENVIRONMENT=production`, `CLINICIAN_INVITE_CODE=<long random>`, `CORS_ORIGINS=["https://<your-frontend>"]`, Supabase keys via env only, Supabase email confirmation ON, and a test project (not production) for `scripts/`.
