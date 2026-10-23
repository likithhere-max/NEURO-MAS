# What's real vs. simulated in NEURO-MAS

Per the brief's explicit "do not fake features" rule, this is a direct accounting of every place
this prototype takes a shortcut, and what a production version would need instead. Nothing below is
hidden in the code either — search for `PROTOTYPE`, `SIMULATED`, `MODEL_NOT_AVAILABLE`, and
`GUIDELINE_REQUIRED` across the codebase to find these same flags at the source.

| Component | What's real | What's simulated/limited | Production replacement |
|---|---|---|---|
| Multi-agent orchestration | Real: typed messages, async execution with per-agent timeout, shared state, fail-closed emergency path, failed agents surfaced as `degraded` | **Stages run sequentially**, not in parallel. Consensus is currently voted by the diagnostic agent only, so it is not a multi-agent agreement measure. Weights unvalidated | Concurrent independent agents, calibrated per-hypothesis support from several agents |
| Diagnostic matching | Real: retrieval/lexical (+ optional Gemini embedding) matching against a curated 10-condition KB; confidence capped | Not a trained classifier; no labelled evaluation | A validated clinical NLP/ML diagnostic support model |
| Signal analysis (tremor/gait) | Real: arithmetic threshold checks on provided summary metrics | No real EEG/EMG/video/voice model — returns `MODEL_NOT_AVAILABLE` explicitly rather than a fake prediction | Trained, validated signal-processing models per modality, each independently evaluated |
| Risk scores | Real: transparent weighted-factor computation | `heuristic_score`: weights not derived from outcome data. Reported confidence is **input coverage**, not probability | Calibration against real outcome data |
| Medication safety | Real: brand→generic + class normalisation, allergy cross-check, 5-rule table, explicit list of medications it could NOT assess | The 5 rules are unsourced illustrations. "No interaction found" does not mean safe | Licensed interaction database |
| Knowledge base / RAG | Real: source-tagged curated summaries with URLs; lexical retrieval, optional Gemini embeddings + pgvector | 10 conditions fully populated; citation liveness unchecked; embedding path unverified in the audit | Clinician-reviewed KB, link checking, evaluated retrieval |
| LLM usage | Real: Google Gemini (`google-genai`) for symptom extraction, narration, chat and embeddings, with timeout, transient-only retry and template fallback. Gemini never makes safety decisions | **Patient free text is sent to Google.** No redaction/consent flow; vendor terms for your key not verified. Prompt-injection resistance of the live model untested | Consent + redaction + DPA, or an on-prem/BAA-covered model |
| Emergency detection | Real: deterministic regex-family phrase matching with narrow negation, run independently of the agent (and on chat messages); fail-closed | Recall on real patient language **unmeasured** (17/17 on my own test set); English only; no structured vital-sign triggers | Clinician-labelled test set, structured triggers |
| Action execution | Actions are proposed, safety-checked and persisted with an audit trail. **"executed" now means a `NotificationProvider` confirmed delivery** | **No provider is implemented**, so nothing is ever sent; emergency actions show as `proposed · NOT_DELIVERED`. Clinician approval still stores `executed` (see OUTSTANDING O15) | Implement SMS/push/pager provider with delivery receipts |
| FHIR | Not implemented in this build | — | Add a FHIR adapter layer mapping `Patient`/`Condition`/`Observation`/`MedicationRequest` to FHIR R4 resources (see `docs/ROADMAP.md`) |
| Auth | Real: Supabase Auth, ES256 JWT verification (exp/sub/issuer required), role guards, clinician invite code, auth rate limiting | No email verification, no MFA; token in browser storage; rate limit per process | Email confirmation, MFA, httpOnly cookies, shared rate-limit store |
| Database | Real: Supabase Postgres with RLS (policies are **not in this repo**; isolation unverified) | Multi-step writes are not transactional. The old SQLAlchemy/SQLite layer was removed | Export schema + RLS into the repo, cross-tenant tests, atomic RPC writes |
| ML demo models | Two scikit-learn demo models, SHA-256 verified before loading | Training selected winners on the evaluation folds; no real-data re-validation done (BLOCKED); not for clinical use | Run `ml/validation/validate_models.py`, nested validation, external cohort |
| Clinical validation | None | Every disease record explicitly carries `"review_status": "PROTOTYPE - pending clinician verification before any clinical use"` | Clinician review of every knowledge-base entry before any real patient interaction |

## Explicit non-claims

This system does not claim, anywhere in its code, UI copy, or this documentation:
- HIPAA/GDPR compliance (only "HIPAA/GDPR-ready design considerations")
- Clinical validation of any score, risk formula, or diagnostic output
- Equivalence between consumer wearable signals and clinical-grade EEG/EMG
- That any citation was machine-generated — every citation traces to a URL in
  `backend/data/diseases_curated.json`
