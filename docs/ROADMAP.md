# NEURO-MAS Roadmap

Tracks the brief's own MUST HAVE / SHOULD HAVE / FUTURE split (its section 29) against what this
session actually built, so a follow-up session (or you, reading this later) knows exactly where to
pick up.

## MUST HAVE — status

| Item | Status |
|---|---|
| Multi-agent orchestration | **Done (prototype)** — 10 registered agents (9 scheduled), dependency-graph orchestrator with deadlines and typed outcomes, shared upstream view |
| Neurological disease knowledge | **Done (partial)** — 10 conditions fully curated and source-tagged; 356 more seed rows in `disease_seed_275.csv` are names-only, pending population |
| Patient profile | **Done** — `PatientState` Pydantic model + Supabase tables (RLS isolation unverified) |
| Personalized assessment | **Done** — Personalization Agent output is provably patient-specific (see `personalization_reasons`) |
| Agent collaboration | **Done** — structured `AgentMessage`s, shared upstream results, visible in UI |
| Consensus | **Done as an uncalibrated index** — typed-evidence ranking, contradiction, ties; not a probability; one rankable voter in practice (`docs/phase2/CONSENSUS_DESIGN.md`) |
| Risk assessment | **Done** — heuristic, explicitly labeled as such |
| RAG | **Done (lexical only)** — real retrieval, real citations, not embedding-based yet |
| Alerts | **Done** — Alert Center, acknowledge flow |
| Audit trail | **Done** — every agent run, consensus computation, and action proposal logged as audit events |
| Working UI | **Done** — patient dashboard showing agent activity, consensus, personalization, actions, alerts, timeline |
| FastAPI backend | **Done** — full route set per the brief's API sketch |
| Database | **Done** — Supabase Postgres (the SQLAlchemy/SQLite layer was removed); SQL migrations in `backend/scripts/` are unexecuted |

## SHOULD HAVE — status

| Item | Status |
|---|---|
| Longitudinal analytics | **Partial** — within-person statistical analysis exists and was characterised by simulation (`docs/phase2/MATHEMATICAL_VALIDATION.md`); no real-data evaluation; no charting yet |
| Wearable ingestion | **Partial** — accepts summary metrics (tremor amplitude, gait speed); no raw-stream ingestion pipeline |
| FHIR adapter | **Not started** — see "Next steps" below |
| Graph knowledge (Neo4j) | **Not started** — deliberately deferred; the brief itself says not to add it unless it "genuinely improves the project," and a 10-condition KB doesn't yet need graph traversal over a relational/JSON store |
| Real-time WebSocket events | **Not started** — the orchestrator currently runs synchronously and returns the full trace at once; the UI's timeline is a real (not fabricated) event log, just not pushed live |

## FUTURE / RESEARCH — status

All explicitly out of scope for this build, per the brief's own instruction: clinical-grade EEG
models, hospital EHR integration, production emergency dispatch, real medication optimization,
FDA/medical-device validation, large-scale prospective clinical validation. None of these are faked
anywhere in this codebase (see `docs/SIMULATED_VS_REAL.md`).

## Concrete next steps, in priority order

1. **Populate the remaining knowledge base rows.** Run `backend/data/build_from_orphadata.py`
   against the free Orphadata product feeds (network access required; not run in this session) to
   pull phenotype/gene/epidemiology data for the other ~356 seed diseases, then clinician-review
   before merging into `diseases_curated.json`.
2. **Swap lexical RAG for embedding-based retrieval.** Add an `EmbeddingProvider` interface
   (mirroring `LLMProvider`), embed the knowledge base with a real embedding model, and use
   pgvector or Qdrant for retrieval. The `KnowledgeBaseService` contract (`search`,
   `match_symptoms`) shouldn't need to change shape, just its internals.
3. **Real-time workflow streaming.** Add a WebSocket endpoint that emits the same `AgentMessage`
   events the orchestrator already generates, as they happen, instead of only after the workflow
   completes. The frontend `AgentActivityPanel` is already structured to accept a stream.
4. **FHIR adapter layer.** Map `Patient`, `Condition`, `Observation`, `Medication` to FHIR R4
   resources; keep it as an adapter (translation layer) rather than restructuring the internal models
   around FHIR, since the internal models serve the agent pipeline first.
5. **Notification provider.** Wire `caregiver_notification` / `clinician_notification` actions to a
   real channel (email/SMS/push) behind a `NotificationProvider` interface, mirroring how
   `LLMProvider` is pluggable.
6. **Auth hardening.** Move off `sessionStorage` token storage before any real patient data is
   involved; add password reset and (for clinician accounts) MFA.
7. **Frontend dependency upgrade.** The frontend currently pins Next.js 15.5.23 / React 18 because a
   jump to Next 16 (which fully resolves the nested `postcss`/`sharp` advisories surfaced by
   `npm audit`) requires a React 19 migration this session didn't have budget to validate end-to-end.
   Track and apply that upgrade before any production deployment.
8. **Test coverage for the frontend.** The backend has 31 passing tests (unit + integration); the
   frontend has none yet. Add component tests for the dashboard panels and an end-to-end test
   (Playwright) covering login → create patient → trigger workflow → approve action.
