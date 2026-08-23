# What's real vs. simulated in NEURO-MAS

Per the brief's explicit "do not fake features" rule, this is a direct accounting of every place
this prototype takes a shortcut, and what a production version would need instead. Nothing below is
hidden in the code either — search for `PROTOTYPE`, `SIMULATED`, `MODEL_NOT_AVAILABLE`, and
`GUIDELINE_REQUIRED` across the codebase to find these same flags at the source.

| Component | What's real | What's simulated/limited | Production replacement |
|---|---|---|---|
| Multi-agent orchestration | Real: typed messages, real async execution, real shared state, real consensus math | — | Add distributed task queue (Celery/Temporal) if agents need to scale independently |
| Diagnostic matching | Real: lexical overlap scoring against real curated data, capped confidence | Not a trained classifier — no ML model backs this | A validated clinical NLP/ML diagnostic support model, trained and clinically evaluated |
| Signal analysis (tremor/gait) | Real: arithmetic threshold checks on provided summary metrics | No real EEG/EMG/video/voice model — returns `MODEL_NOT_AVAILABLE` explicitly rather than a fake prediction | Trained, validated signal-processing models per modality, each independently evaluated |
| Risk scores | Real: transparent weighted-factor computation, fully inspectable | Explicitly labeled `heuristic_score`, not `validated_clinical_risk_score` — weights are not derived from outcome data | Calibration against real outcome data, ideally a published/validated risk score (e.g. a seizure-recurrence risk tool) |
| Medication safety | Real: deterministic table lookup, real allergy cross-check | The interaction table is a small illustrative set (5 rules), not an exhaustive drug database | Integrate a licensed interaction database (e.g. a First Databank-style feed) |
| Knowledge base / RAG | Real: source-tagged, curated summaries with URLs and `last_verified` dates; real lexical retrieval | Only 10 conditions have full detail; retrieval is keyword-overlap, not embedding-based semantic search; ICD-11 and Orphanet codes are marked `PENDING_VERIFICATION` rather than guessed | Populate the remaining ~360 seed rows via `build_from_orphadata.py` + guideline review; swap in an embedding model + pgvector/Qdrant |
| LLM usage | Real: pluggable `LLMProvider` interface | Default provider is a deterministic template — no external model call happens unless `ANTHROPIC_API_KEY` is set, and even then the wiring is not live in this prototype (see `services/llm_provider.py`) | Wire the Anthropic (or other) call with real timeout/retry/fallback handling for the Patient Copilot's narrative phrasing only — never for safety-critical decisions |
| Emergency detection | Real: deterministic phrase-matching, unit-tested | Only free-text substring matching — no structured vital-sign or wearable-triggered detection yet | Add threshold-based triggers on structured vitals/wearable streams, not just text |
| Action execution | Real: actions are proposed, safety-checked, and persisted with full audit trail | "Executed" for low-risk actions means the action record is marked executed — no real push notification, SMS, or pager integration fires | Wire a `NotificationProvider` (e.g. Twilio/SMS, push, pager) behind the same interface |
| FHIR | Not implemented in this build | — | Add a FHIR adapter layer mapping `Patient`/`Condition`/`Observation`/`MedicationRequest` to FHIR R4 resources (see `docs/ROADMAP.md`) |
| Auth | Real: JWT + bcrypt password hashing + role-based route guards | Token stored in `sessionStorage` (prototype convenience) rather than httpOnly cookies; no MFA, no password reset flow | Move to httpOnly session cookies or a managed auth provider before any real patient data touches this |
| Database | Real: SQLAlchemy models, real persistence, portable to Postgres | Ships on SQLite by default (no external infra needed for a student demo) | Point `DATABASE_URL` at Postgres; add Alembic migrations |
| Clinical validation | None | Every disease record explicitly carries `"review_status": "PROTOTYPE - pending clinician verification before any clinical use"` | Clinician review of every knowledge-base entry before any real patient interaction |

## Explicit non-claims

This system does not claim, anywhere in its code, UI copy, or this documentation:
- HIPAA/GDPR compliance (only "HIPAA/GDPR-ready design considerations")
- Clinical validation of any score, risk formula, or diagnostic output
- Equivalence between consumer wearable signals and clinical-grade EEG/EMG
- That any citation was machine-generated — every citation traces to a URL in
  `backend/data/diseases_curated.json`
