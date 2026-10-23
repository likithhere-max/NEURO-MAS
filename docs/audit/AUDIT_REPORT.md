# NEURO-MAS — Repository Audit Report (Deliverable 1)

> **Point-in-time record (Phase 1 audit, 2026-10-23, commit `1be5e52`).** Parts of this document have been superseded by Phase 2 — see `docs/phase2/IMPLEMENTATION_PLAN.md` and `docs/phase2/OUTSTANDING_ISSUES.md`. Where the two disagree, **the Phase 2 document is current**. Inline notes marked **[Phase 2]** flag changed items.

Audit date: 2026-10-23. Scope: backend, ML artifacts, agents, security, frontend↔backend integration.
Method: read the existing code, reproduce each suspected defect with a script **against the original
code first**, fix it, then re-run. Nothing was rebuilt; the public API shape, the DB schema and the
dependency set were kept except where listed in `CHANGE_SUMMARY.md`.

**Status vocabulary** — FIXED (code changed + test passes) · PARTIAL · OPEN · BLOCKED (cannot be
verified from this environment; reason given) · NOT IMPLEMENTED.

**Honesty note.** "FIXED" means "the defect I reproduced no longer reproduces and a regression test
guards it". It does not mean clinically validated. Several fixes were tested only against data I wrote
myself (flagged below). Real-data ML validation was BLOCKED (see `ML_VALIDATION_REPORT.md`).

---

## 1. Findings by severity

### CRITICAL

| ID | Finding | Where | Root cause → Impact | Fix | Verification | Status |
|---|---|---|---|---|---|---|
| C1 | Anyone could self-register as a **clinician** | `api/routes/auth.py::register` | Role came straight from the request body. A clinician can approve restricted actions and read patient data → privilege escalation by anyone on the internet. | `_check_clinician_signup_allowed`: clinician signup needs `CLINICIAN_INVITE_CODE` (constant-time compare); with no code configured it is allowed only when `ENVIRONMENT=development`. Rate-limited, generic errors. | `test_security.py` (invite required / wrong code / dev fallback / prod closed) | FIXED. **Residual:** signup still uses `email_confirm=True` (no email verification). **Deployment action required:** set `ENVIRONMENT=production` and `CLINICIAN_INVITE_CODE`. |
| C2 | Emergency actions were recorded as **"executed"** although nothing was ever sent | `core/actions.py::build_action` | No delivery mechanism existed; the status was set by policy alone. UI told a patient/clinician an escalation had happened. Patient-safety misrepresentation. | `services/notifier.py` `NotificationProvider`; an action is `executed` only if a provider returns `delivered=True`. Default `NullNotificationProvider` → status `proposed`, reason `NOT_DELIVERED:…`. UI shows "proposed · not sent". | `test_actions_notifier.py` | FIXED (honesty). **No real channel exists** — Notifier agent NOT IMPLEMENTED. |
| C3 | Emergency detector recall was **1/15** on realistic phrasings | `services/clinical_rules.py` | 7 exact literal phrases; "face is drooping", "can't speak", "I'm having a seizure", curly apostrophes, etc. all missed → fail-open on the safety path. | Regex families (stroke, epilepsy, breathing, GBS, general), normalisation, narrow same-clause negation ("denies chest pain"), `detect_emergency_detailed`. Pre-scan in orchestrator runs independently of the agent. | `test_emergency_rules.py`: 17/17 positives, negation and false-positive cases. | FIXED **on my own test set only**. Independent real-world recall is **unmeasured**; the detector is phrase-matching and will miss novel wording. |
| C4 | The Supabase **schema and RLS policies are not in the repo** | whole data layer | Cannot audit tenant isolation. Only `profiles` RLS was ever tested (`scripts/test_rls_deny.py`, against production, with a hard-coded UUID). | None possible without credentials. | — | OPEN / BLOCKED. Needs `supabase db dump` + a separate test project. |

### HIGH

| ID | Finding | Where | Root cause → Impact | Fix | Verification | Status |
|---|---|---|---|---|---|---|
| H1 | Consensus could not discriminate between diseases | `core/consensus.py` | Every hypothesis had the same voters and `s_i` = the agent's single global confidence → measured scores 0.245 vs 0.245 (a tie). `e_i` also defaulted to confidence → confidence squared. | Hypothesis-specific `s_i(H)`; non-supporting/failed agents **abstain**; `e_i=1` when support is hypothesis-specific; `margin` reported; no voter ⇒ no action. | `test_consensus.py`, hand calc in `MATHEMATICAL_SPECIFICATION.md` | FIXED |
| H2 | Trend direction inverted for "lower-is-worse" metrics | `agents/longitudinal.py`, `repository.py` | A drop in gait speed or MoCA was labelled "improving". | `core/metric_semantics.py` polarity registry + `classify_change`; unknown polarity ⇒ `changed_direction_unknown`, never guessed. | `test_metric_semantics.py` | FIXED |
| H3 | Baseline = 0 produced fake large % changes | same | Division by ~0 → artefact "anomalies". | `baseline_zero_undefined` classification, no percentage. | `test_metric_semantics.py` | FIXED |
| H4 | **Silent agent failure**: `symptoms=None` crashed an agent yet the workflow reported "completed" | `orchestrator.py` | Exceptions swallowed; no surfaced degradation. | `failed_agents`, `degraded`, per-agent `asyncio.wait_for` timeout, shown in API and UI banners. | `test_orchestrator.py`, `test_workflow_routes.py` | FIXED |
| H5 | **Fail-open emergency path**: crash of the emergency agent ⇒ `is_emergency=False` | `orchestrator.py` | Safety function failing → "no emergency". | Independent deterministic pre-scan; failed emergency agent forces `is_emergency=True` (`failed_closed`); crash of pre-scan also fails closed. | `test_orchestrator.py` | FIXED |
| H6 | Patient chat bypassed emergency screening | `services/chat_assistant.py` | New messages went to the LLM or to a "no assessment yet" reply even if they described a stroke. | Deterministic scan of every new message **before** anything else; fixed `LIVE_EMERGENCY_REPLY`, LLM not called. | `test_chat_assistant.py` | FIXED |
| H7 | Chat: no input cap, no prompt-injection delimiting, no output guardrail | same + `schemas/chat.py` | Unbounded cost/abuse; user text could override instructions; model could emit doses/diagnoses. | 2000-char cap; `<patient_message>` delimiting; regex output guardrail (`_DOSE_RE`, `_DIAGNOSIS_RE`). | `test_chat_assistant.py` (offline). **Live Gemini injection resistance NOT tested.** | PARTIAL |
| H8 | Fabricated "confidence" numbers | `risk.py`, `longitudinal.py`, `knowledge.py`, `medication.py`, `data_intake.py` | Hard-coded or arbitrary values displayed as if they were probabilities. | Confidence redefined as *coverage* (inputs/metrics/KB/table) and labelled so (`confidence_is_*_not_probability` warnings; UI shows "coverage"). | `test_agents.py`, `test_input_validation.py` | FIXED (relabelled; not calibrated) |
| H9 | Medication check missed brand names, drug classes, and unknown drugs; unknown drugs looked "safe" | `clinical_rules.py`, `medication.py` | Substring match on generics only. | Brand→generic map, class members (aminoglycoside, penicillin), looser allergy matching, `interaction_check_incomplete` for unassessed drugs. | `test_drug_rules.py` | FIXED for coverage honesty. **The rule table is still 5 unsourced rules** — NOT a drug-interaction database. |
| H10 | Adherence unit hole (0–100 vs 0–1) | `medication.py`, `schemas/api.py` | 85 interpreted as 8500% / 0.85 ambiguity. | Schema: 0–1 finite; agent validates and warns. | `test_input_validation.py` | FIXED |
| H11 | Anomaly detection used a fixed z-threshold | `ml/personalized_anomaly.py` | With n=3–5 readings a fixed |z|≥2 gives far more than the nominal false-alarm rate (Monte-Carlo in the maths spec). | Student-t prediction-interval threshold `t_{0.975,n-1}·√(1+1/n)`; non-finite values dropped; zero-variance tolerance. | `test_personalized_anomaly.py`; simulation: FPR ≈ 0.05 at n=3,5,10,30 | FIXED (under Gaussian, i.i.d. assumptions — real PD data are neither) |
| H12 | Input validation: NaN/inf/negative/implausible values accepted; NaN → HTTP 500 | `schemas/api.py`, `data_intake.py`, `main.py` | Poisoned state, crash. | Pydantic constraints, `allow_inf_nan=False`, plausible-range table, de-duplication, custom 422 handler. | `test_input_validation.py` | PARTIAL (plausible ranges are my engineering judgement, not clinical reference ranges) |
| H13 | Signal-analysis agent crashed on missing/zero baseline and claimed analyses it does not perform | `signal_analysis.py` | — | `SUPPORTED_SUMMARY_METRICS` vs `RECORDED_ONLY_METRICS`; guards; truthful summary. | `test_agents.py` | FIXED |
| H14 | Diagnostic query polluted by a placeholder symptom string | `diagnostic.py` | Retrieval matched on the literal placeholder. | Placeholder excluded; real name **and** free text used. | `test_agents.py` | FIXED |
| H15 | Alert acknowledge / action approve had no role check; approve could re-approve any row | `api/routes/alerts.py` | Patient could approve own restricted action. | Clinician-only; approve only when `status=="pending_approval"` (404 otherwise). | `test_security.py` | FIXED. Approve still writes status `executed` although nothing is sent (kept for UI/DB compatibility) — see OUTSTANDING M4. |
| H16 | Patient→clinician assignment not validated | `api/routes/patients.py` | A patient could be attached to any UUID. | `_resolve_clinician_id` with admin-client existence check. | `test_ml_demo_and_patients.py` | FIXED |
| H17 | Auth: shared cached "public" Supabase client reused across requests | `api/routes/auth.py`, `core/supabase_client.py` | `sign_in` mutates client session → possible identity bleed between concurrent requests (found by code review; not reproduced against live Supabase). | `new_public_client()` per request; authed client per token. | Code review only (no live-Supabase reproduction) | FIXED on code-review basis |
| H18 | JWT verification did not require `exp`/`sub` or check issuer | `services/auth.py` | Tokens without expiry accepted. | `options={"require":["exp","sub"]}`, issuer check. | `test_auth_jwt.py` (self-signed ES256 key) | FIXED. **Not tested against real Supabase JWKS.** |
| H19 | No rate limiting on auth | `api/routes/auth.py` | Brute force / signup spam. | In-memory sliding window, 10/min/IP. | `test_security.py` | PARTIAL: per process only; not shared across workers/instances, trusts `request.client.host`. |
| H20 | **PHI/PII is sent to a third-party LLM (Google Gemini)** | `ai/gemini_service.py`, chat, symptom extraction, embeddings | Patient free text leaves the system. Free-tier API terms may permit data use for product improvement (not verified). No consent/DPDP/HIPAA disclosure in the app. | Documented in `SECURITY_PRIVACY_REPORT.md`; `SIMULATED_VS_REAL.md` updated. No technical mitigation (no redaction). | — | OPEN |
| H21 | Gemini calls: no timeout, retry on any exception | `ai/gemini_service.py` | Hung requests; retry storms on 4xx. | 30 s timeout; retry only transient errors (408/429/5xx/transport). | `test_integrations_hardening.py` | FIXED. Worst-case latency of the serial pipeline with several 30 s calls remains OPEN. |
| H22 | Non-atomic persistence of a workflow | `api/routes/workflows.py`, `repository.py` | Partial writes possible; an emergency could be lost if DB write failed. | `_persist_or_degrade`: emergency result is still returned with `persisted:false`; non-emergency ⇒ clean 500. | `test_workflow_routes.py` | PARTIAL — still multiple separate inserts, no DB transaction (Supabase REST has none). |
| H23 | ML artefacts loaded with `joblib.load` (pickle ⇒ arbitrary code execution) with no integrity check | `api/routes/ml_demo.py` | A swapped `.joblib` = RCE. | SHA-256 verified against `ml/models/MANIFEST.json` before load; mismatch ⇒ 503. | `test_ml_demo_and_patients.py` | FIXED (manifest is in the same repo → protects against accidental/partial tampering, not against an attacker who can edit both). |

### MEDIUM

| ID | Finding | Status |
|---|---|---|
| M1 | **ML models trained with selection bias**: hyper-parameters/model chosen on the same LOSO folds that report performance; no CI, no calibration, no dummy baseline, no subject-level aggregation. Any previously quoted headline metric from that procedure should not be trusted. | Tooling delivered (`ml/validation/validate_models.py`: nested LOSO, subject bootstrap CI, Brier/ECE, dummy baseline, permutation test, prevalence-adjusted PPV/NPV). **Real results BLOCKED** — UCI/HuggingFace unreachable from this sandbox. Run it on your machine. |
| M2 | Model artefacts: sklearn version mismatch (trained 1.9.0 vs pinned 1.9.1), no version/hash metadata | Manifest + requirement pin `>=1.9.0,<1.10` added. Retrain recommended. |
| M3 | Risk score is a heuristic with unvalidated weights; consensus reliability weights are informal | Labelled as such; OPEN (needs outcome data). |
| M4 | "Approved" status semantics: approving writes `executed` although nothing is sent | OPEN (needs DB check-constraint knowledge; schema not in repo). |
| M5 | Knowledge agent only does URL lookup/retrieval; citation liveness/accuracy unchecked; `last_verified`/`review_status` may be absent | UI made defensive; liveness unchecked. |
| M6 | No idempotency on `POST /workflows` (double-click ⇒ duplicate workflows/alerts) | OPEN |
| M7 | Docstring claimed agents run "in parallel"; the orchestrator is sequential (9 stages) | Doc corrected; latency = sum of agents. |
| M8 | `scripts/test_rls_deny.py` runs against production Supabase with hard-coded UUID; the shared test-account password (literal omitted; it remains in git history) hard-coded in 7 scripts of a **public** repo; test account `apitest@example.com` exists | OPEN — rotate/disable the test account(s); move creds to env. |
| M9 | CORS: `"*"` default combined with `allow_credentials=True` | FIXED: credentials only with explicit origins; warning in non-dev. |
| M10 | Dead SQLAlchemy scaffolding (`app/db`, `app/seed`), `create_all` on import, unused/vulnerable deps (python-jose, passlib, bcrypt…) and committed SQLite test DB | Removed; `requirements.txt` rewritten (see Change Summary). |

### LOW

| ID | Finding | Status |
|---|---|---|
| L1 | Copilot agent registered but never invoked by the orchestrator | OPEN (documented) |
| L2 | Stale documentation (README, `SIMULATED_VS_REAL.md`, `ARCHITECTURE.md` §7) | FIXED in this change |
| L3 | Coverage of route modules is low (`patients.py` 32%, `chat.py` 36%, `repository.py` 58%) because they need live Supabase | OPEN |
| L4 | Stray `.coverage`, `tsbuildinfo` artefacts | FIXED (`.gitignore`) |

---

## 2. What could **not** be verified (do not read these as passing)

1. Supabase schema, RLS for all tables, cross-tenant isolation (credentials/schema absent).
2. Live Gemini behaviour: prompt-injection resistance, embedding/pgvector path, latency.
3. Real email verification, real notification delivery.
4. Any real-data ML metric (datasets unreachable).
5. KB citation URLs are live and say what the KB claims.
6. Independent emergency-detector recall.
7. Gemini data-handling terms for the account actually in use.

## 3. Headline numbers (measured this session)

* `pytest tests --cov=app`: **226 passed, 4 skipped**, 81 % line coverage (the 4 skipped need live Supabase: `NEUROMAS_LIVE_TESTS=1`).
* Mutation sanity check: against the **original** `app/` source, 56 of the new tests fail and 3 test files cannot import → the tests do detect the defects.
* Ruff (F, E9) clean. Frontend `tsc --noEmit` and `next build` pass.
