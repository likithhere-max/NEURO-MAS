# What's real vs. simulated in NEURO-MAS

Per the brief's explicit "do not fake features" rule, this is a direct accounting of every place
this prototype takes a shortcut, and what a production version would need instead. Nothing below is
hidden in the code either — search for `PROTOTYPE`, `SIMULATED`, `MODEL_NOT_AVAILABLE`, and
`GUIDELINE_REQUIRED` across the codebase to find these same flags at the source.

| Component | What's real | What's simulated/limited | Production replacement |
|---|---|---|---|
| Multi-agent orchestration | Typed messages; agents scheduled as a dependency graph with bounded concurrency, per-agent timeouts inside a global deadline, cancellation, typed `timed_out`/`abstained`/`insufficient_data` outcomes, fail-closed emergency, emergency pre-scan checkpointed before slow agents; verified with **fake** slow/hung/failing agents | Not load-tested; real-agent latency unmeasured | Load/soak tests, tracing |
| Consensus | Typed-evidence support **index** (no probability), independence groups, contradiction, ties → no leader | Only the diagnostic agent supplies rankable evidence in practice; weights are informal; nothing is calibrated | Labelled outcomes → per-voter validation → calibrated stacking |
| Diagnostic matching | Retrieval/lexical (+ optional Gemini embedding) matching against a curated 10-condition KB | Not a trained classifier; no labelled evaluation | Validated clinical decision-support model |
| Longitudinal analysis | Real statistics (t-interval, Theil–Sen, Mann–Kendall, self-starting CUSUM/EWMA) with an autocorrelation gate; behaviour **measured by simulation** (false-alarm 7–43 % over a monitoring course depending on noise) | Provisional thresholds; no real patient series ever analysed; no MCID unless a clinician-approved file is supplied | Real longitudinal data; clinician-set MCIDs; alarm-budget policy |
| Signal analysis (tremor/gait) | Arithmetic threshold checks on provided summary metrics | No EEG/EMG/video/voice model: status `abstained` instead of a fake prediction | Trained, validated signal-processing models |
| Risk scores | Transparent weighted-factor computation | `heuristic_score`: weights not derived from outcome data; "confidence" is input coverage | Calibration against outcome data |
| Medication safety | Brand→generic + class normalisation, allergy cross-check, `DDISource` interface with seven distinct statuses, explicit list of drugs it could not assess | **No interaction database is wired up**; the 5-rule table can only report "combination not assessed"; "no interaction found" never means safe | Licensed/open DDI source (or versioned snapshot) + pharmacist review |
| Knowledge base / RAG | Source-tagged curated summaries with URLs; lexical retrieval, optional Gemini embeddings + pgvector; link checker and review queue exist | 10 conditions populated; URL liveness **could not be checked** from the build environment; codes pending clinician verification | Clinician review; periodic link checks |
| LLM usage | Gemini for symptom extraction, narration, chat and embeddings, with timeout, transient-only retry and template fallback; never makes safety decisions; **consent gate** (per-patient flag, default off) and best-effort redaction, `EXTERNAL_PROCESSING_MODE` switch | Redaction is not de-identification; consent column needs the privacy migration (unexecuted); legal basis not assessed | Provider agreement/DPA, legal review, de-identification pipeline |
| Emergency detection | Deterministic regex-family phrase matching with narrow negation, independent of the agent and applied to chat; fail-closed | Recall on real patient language unmeasured; English only | Independently authored multilingual test set; structured triggers |
| Action execution / notifications | Outbox with explicit states (proposed → approved → queued → sent → delivered → acknowledged / failed → escalated), idempotency, bounded retries, escalation, PHI-free messages; "executed" only after confirmed delivery | Default channel `none`; the SMTP provider has only been exercised against a stand-in for `smtplib`; **nothing has ever been sent**; approve-route dispatch and manual re-drive not implemented | Provider + credentials + on-call policy, staged delivery tests |
| FHIR | Not implemented | — | FHIR R4 adapter layer (see `docs/ROADMAP.md`) |
| Auth | Supabase Auth, ES256 JWT verification, role guards, clinician invite code; e-mail confirmation no longer bypassed; per-IP auth and per-user workflow/chat rate limits (optional Redis, trusted-proxy aware) | MFA not implemented; Supabase "Confirm email" setting unverifiable here; token in `sessionStorage`; Redis path tested only through a fake | MFA (`aal2`), HttpOnly-cookie session, edge rate limiting |
| Database | Supabase Postgres with RLS (policies **not in this repo**); idempotent workflow ids; atomic persistence RPC available behind `PERSIST_VIA_RPC` | SQL migrations **never executed**; RLS isolation **unverified** (live matrix written but skipped) | Export schema + RLS; run the matrix on a test project; apply migrations |
| ML demo models | Two scikit-learn demo models; SHA-256 check + restricted unpickler before loading; OOD/abstention returns no prediction for foreign inputs | Selection was done on the evaluation folds; **no real-data re-validation (BLOCKED)**; artifacts trained under scikit-learn 1.9.0 load with version warnings; not for clinical use | Run the Phase 2 protocol; re-train under the pinned version; non-pickle format |
| Clinical validation | None | Every disease record carries `"review_status": "PROTOTYPE - pending clinician verification before any clinical use"` | Clinician review of every knowledge-base entry before any real patient interaction |

## Explicit non-claims

This system does not claim, anywhere in its code, UI copy, or this documentation:
- HIPAA/GDPR compliance (only "HIPAA/GDPR-ready design considerations")
- Clinical validation of any score, risk formula, or diagnostic output
- Equivalence between consumer wearable signals and clinical-grade EEG/EMG
- That any citation was machine-generated — every citation traces to a URL in
  `backend/data/diseases_curated.json`
