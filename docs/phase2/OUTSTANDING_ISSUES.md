# Outstanding Issues (Phase 2)

Supersedes `docs/audit/OUTSTANDING_ISSUES.md`. "Needs you" = requires credentials, data, a decision or authorization I do
not have. Nothing here is a claim of production readiness; **the system must not be used with real patients.**

## A. Blockers before any real-patient use
| ID | Issue | Why open | Needs you |
|---|---|---|---|
| B1 | **Tenant isolation unverified.** RLS policies/schema are not in the repo; the live RLS matrix has never run | no Supabase project/credentials | export schema+policies; create a **separate test project**; apply the 3 migrations there; run `pytest tests/rls_live`; fix failures; repeat for production |
| B2 | **Exposed credential.** Shared test password literal in 7 scripts in public git history since 2026-09-09; test e-mail addresses remain in scripts | cannot revoke accounts from here | disable/rotate those accounts and any account reusing the password; decide whether to rewrite history (your explicit call; invalidates clones) |
| B3 | **Nothing pushed.** Phase 1 push returned 403 and Phase 2 commits are local only (`phase2/advanced-ml-hardening`, on top of `audit/neuro-mas-hardening`) | no write access from the session | link GitHub / push yourself after reviewing |
| B4 | **SQL never executed** (persistence/RPC, privacy/consent, notification outbox) | no DB access | review, adapt column types to your schema, apply to the test project first |
| B5 | **No live notification channel verified.** SMTP provider tested against a `smtplib` stand-in only; default channel `none` | needs provider, credentials, policy on who is paged | choose channel; test in staging with a consenting recipient; define on-call/escalation policy |
| B6 | **External LLM processing**: consent gate and best-effort redaction exist; the legal basis, provider terms and DPA do not | legal/product decisions | review provider terms for your key/plan; decide default mode (`disabled` is the safest); consent UX |
| B7 | **Email confirmation / MFA**: signup now needs confirmation, but the Supabase "Confirm email" setting is unverifiable here; MFA not implemented | project setting + product decision | enable confirmation; decide `aal2` requirement for clinicians (SECURITY_AND_PRIVACY.md S4) |
| B8 | **CORS** default list still contains `"*"` | deployment-specific | set `CORS_ORIGINS` |

## B. Scientific validity (all models/thresholds remain CLINICALLY UNVALIDATED)
| ID | Issue | Needs you |
|---|---|---|
| V1 | **Real-data ML validation NOT RUN** (dataset download blocked); shipped metadata numbers are from the optimistic procedure; artifacts trained under scikit-learn 1.9.0 load with version warnings under 1.9.1; metadata lacks sklearn version/data hash | run ML_EXPERIMENT_PROTOCOL.md §10 on a networked machine; re-train under the pinned version; regenerate `MANIFEST.json` |
| V2 | OOD domain for the demos is derived from scaler moments (provisional); false-reject/missed-OOD rates unknown | fit a domain from real data (`fit_domain`) and run `evaluate_domain_detector` |
| V3 | Longitudinal detector false alarms on pure noise are 7–43 % over a monitoring course depending on autocorrelation/heavy tails (MATHEMATICAL_VALIDATION.md); no multiplicity control across metrics; never evaluated on real series | real longitudinal data; decide acceptable alarm burden with clinicians; consider per-patient alarm budgets / state-space models |
| V4 | Change thresholds (5 % / 20 %), risk-heuristic weights, consensus reliability weights, plausible ranges, signal 1.5×/0.8× thresholds: engineering defaults | clinician-defined MCIDs (use `METRIC_MCID_PATH` with `approved_by` + `source`) and outcome data |
| V5 | Diagnostic agent = retrieval over 10 conditions, no labelled evaluation; only one agent votes, so "consensus" is not multi-agent agreement | labelled vignettes + clinician review; define validated per-hypothesis evidence for other agents; then calibration (CONSENSUS_DESIGN.md §6) |
| V6 | Emergency detector recall on real patient language unmeasured; English only; no structured vital-sign triggers | ≥ 200 independently authored, de-identified phrases (multilingual); structured triggers |
| V7 | DDI: **no database wired up**; the 5-rule table can only say "combination not assessed" | integrate a licensed/open interaction source behind `DDISource` (or supply a versioned snapshot via `DDI_SNAPSHOT_PATH`) and record its version/licence; pharmacist review |
| V8 | KB citations: liveness unchecked (all 23 URLs unreachable from the sandbox); ICD/Orphanet codes pending verification | run `scripts/check_kb_links.py` with internet; clinician spot-review via `data/kb_review_queue.json` |

## C. Engineering
| ID | Issue |
|---|---|
| E1 | Approve-route does not dispatch the notification; there is no manual re-drive method for `failed`/`escalated` records |
| E2 | Orchestrator concurrency/deadlines tested only with fake agents; no load test; real-agent latency unmeasured |
| E3 | Rate limiting: Redis path tested through a fake; limiter fails open to per-process memory if Redis errors; no edge limiter |
| E4 | Pickle remains the model format; restricted loader is defence in depth only (prefer ONNX/skops after re-training) |
| E5 | `requirements.lock.txt` is a `pip freeze` of the test venv (includes a few non-runtime packages), not hash-pinned; no `pip-audit` run |
| E6 | Frontend: token in `sessionStorage` (XSS-exposed); E2E uses a mocked API; Chromium only; no automated accessibility audit; some pages (patient tabs) have no tests |
| E7 | Low-coverage modules: `routes/patients.py` 36 %, `routes/chat.py` 37 %, `llm_provider.py` 35 %, `symptom_extraction.py` 35 %, `supabase_client.py` 38 % |
| E8 | `copilot` agent registered but unused; treatment-support role intentionally not implemented |
| E9 | Secret scanner covers common key shapes and quoted password literals only, committed content only |
| E10 | Live scripts in `backend/scripts/` still contain test e-mail addresses and were not re-run |

## D. Mapping of Phase 1 issues (O1–O24) to Phase 2 outcome
| Phase 1 | Outcome |
|---|---|
| O1 RLS unverified | **Open** → B1 (matrix written, guards tested, not executed) |
| O2 no notification channel | **Partly addressed**: outbox + SMTP provider + truthful statuses; no live delivery → B5 |
| O3 PHI to Gemini | **Partly addressed**: consent gate, mode switch, redaction → B6 |
| O4 emergency recall unknown | **Open** → V6 |
| O5 e-mail verification / MFA | **Verification: addressed in code** (setting unverifiable); **MFA: evaluated, not implemented** → B7 |
| O6 exposed test credentials | **Working tree cleaned; exposure remains in history** → B2 |
| O7 real-data ML validation | **Open (BLOCKED)** → V1; runner, protocol, guards complete |
| O8 unvalidated weights/thresholds | **Open**; thresholds now configurable with provenance; MCID only if approved → V4 |
| O9 drug table = 5 rules | **Partly addressed**: interface + honest statuses; no database → V7 |
| O10 diagnostic agent unevaluated | **Open** → V5 |
| O11 one voter | **Addressed in design/labelling** (typed evidence, uncalibrated index, non-voting statuses); **still one voter in practice** → V5 |
| O12 Gaussian i.i.d. anomaly | **Addressed in method** (robust/sequential/trend with gates), **characterised by simulation**; real-data evaluation open → V3 |
| O13 no OOD | **Addressed** (abstention); real-data error rates open → V2 |
| O14 KB citations | **Tool built; live check not possible here** → V8 |
| O15 approve writes `executed` | **Addressed**: action statuses `queued / sent_unconfirmed / executed (only after delivered) / delivery_failed`; approve-route dispatch → E1 |
| O16 no idempotency key | **Addressed** (Idempotency-Key → deterministic id; tested with fake DB; SQL index unexecuted) |
| O17 non-transactional persistence | **Addressed in code** (atomic RPC behind a flag; SQL unexecuted) + incomplete-marker fallback |
| O18 serial pipeline / no deadline | **Addressed** (DAG, bounded concurrency, deadlines; fake-agent tests) → E2 |
| O19 per-process rate limit | **Addressed in code** (Redis option, trusted proxy, per-user) → E3 |
| O20 live tests not run / low coverage | **Unchanged** for live; coverage now 86 % overall → E7 |
| O21 copilot unused / no treatment support | **Unchanged**, intentional → E8 |
| O22 frontend tests / token / messages | **Tests and messages addressed**; token storage unchanged → E6 |
| O23 retrain / safe format | **Restricted loader added; retrain open** → V1, E4 |
| O24 evidence-tab citation page | **Not provided; tool for KB links built** → V8 |

## E. Deployment checklist (necessary, not sufficient)
`ENVIRONMENT=production`; `CLINICIAN_INVITE_CODE` set; `CORS_ORIGINS` pinned; keys only in the environment; Supabase email
confirmation ON and MFA decision made; test accounts disabled; migrations applied and RLS matrix passing **on a test project
first**; `EXTERNAL_PROCESSING_MODE` chosen deliberately; `NOTIFICATION_CHANNEL` configured and tested in staging;
`TRUSTED_PROXY_COUNT` set to your real proxy depth; independent security and clinical-safety review completed.
