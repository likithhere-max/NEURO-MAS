# NEURO-MAS — Outstanding Issues (Deliverable 8)

Ordered by what I would do first. "Needs you" = requires credentials, a decision, or data I do not have.

## Blockers for any use with real patients
| # | Issue | Why open | Needs you |
|---|---|---|---|
| O1 | **Supabase schema + RLS not in repo; cross-tenant isolation unverified** (C4) | No credentials/schema in sandbox | `supabase db dump --schema public > supabase/schema.sql`; a separate **test** Supabase project; then add two-user cross-access tests (patient A ↛ patient B; patient ↛ clinician-only routes; unassigned clinician ↛ patient). |
| O2 | **No real notification channel** (C2): emergencies are detected and displayed, never sent | Needs provider + credentials + policy decision (who is paged?) | Pick SMS/email/push; implement `NotificationProvider.deliver`; add delivery receipts and retry. |
| O3 | **PHI sent to Google Gemini** without consent/redaction (H20) | Legal/product decision | Verify Gemini API terms for your key (paid vs free tier), add consent screen, redact identifiers, DPDP notice, DPA. |
| O4 | **Emergency detector real-world recall unknown** (C3) | Needs independently-authored, de-identified test phrases (ideally clinician-written, multilingual) | Provide/label ≥200 phrases; add structured triggers (vitals, wearable) as second channel. |
| O5 | **Email verification absent**; MFA absent | `email_confirm=True` in signup | Turn on Supabase email confirmation and remove the flag. |
| O6 | **Public repo contains `TestPassword123!` and test emails** | Account hygiene | Disable/rotate `apitest@example.com`, `neuromas.test.user*@example.com`; replace with env-driven creds. |

## Scientific / validation
| # | Issue | Needs you |
|---|---|---|
| O7 | **Real-data ML validation not run** (BLOCKED) | Run `python ml/validation/validate_models.py disease|risk|artifacts` locally and paste results into `ML_VALIDATION_REPORT.md` §4. Retrain with nested selection if gates fail. |
| O8 | Risk heuristic weights, consensus reliability weights, change thresholds (5 %/20 %), plausible ranges: all unvalidated engineering defaults | Clinician-defined MCIDs/ranges and outcome data. |
| O9 | Drug table = 5 unsourced rules | Integrate a licensed/open interaction source (e.g. RxNorm + a curated DDI list) and cite versions. |
| O10 | Diagnostic agent = retrieval over 10 conditions; no labelled evaluation | Labelled vignettes + clinician review; report top-k accuracy with CI. |
| O11 | Only the diagnostic agent votes in consensus → "consensus" is not multi-agent agreement | Define per-hypothesis support for other agents (e.g. risk, knowledge) and calibrate. |
| O12 | Anomaly detector assumes Gaussian i.i.d.; real series are autocorrelated/trending/state-dependent | Evaluate on real longitudinal data; consider robust/EWMA/state-space methods. |
| O13 | No OOD detection on ML demo inputs | Add range/Mahalanobis guard + "outside training distribution" response. |
| O14 | KB citations: liveness/accuracy unchecked; ICD/Orphanet codes pending verification | Run a link checker and clinician spot-review. |

## Engineering
| # | Issue |
|---|---|
| O15 | M4 — approving an action writes `executed` though nothing is sent; needs a DB status vocabulary decision (e.g. `approved`). |
| O16 | M6 — no idempotency key on `POST /workflows`; double submission duplicates rows/alerts. |
| O17 | H22 — workflow persistence is several non-transactional inserts; consider a Postgres function (RPC) to write atomically. |
| O18 | H21 — serial pipeline can exceed minutes if several Gemini calls hit the 30 s timeout; add overall deadline and run independent agents concurrently. |
| O19 | H19 — rate limit is per process; use a shared store (Redis) and trusted-proxy handling. |
| O20 | Live tests (`NEUROMAS_LIVE_TESTS=1`) never executed here; low coverage on route/repository modules (32–58 %). |
| O21 | Copilot agent unused; Treatment-Support role not implemented (intentionally — needs clinical governance). |
| O22 | Frontend: no unit/E2E tests; token in browser storage; no 429/validation-friendly messages. |
| O23 | Models: retrain under pinned scikit-learn; consider a safe format (skops/ONNX) instead of pickle. |
| O24 | Evidence-tab citation verification page/script from earlier conversation was never provided. |

## Deployment checklist (must do)
`ENVIRONMENT=production`, `CLINICIAN_INVITE_CODE`, `CORS_ORIGINS` pinned, keys only in env, Supabase email confirmation ON, test accounts removed, `NEUROMAS_LIVE_TESTS` only against a test project.
