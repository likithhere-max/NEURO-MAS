# Agent Capability Matrix (Phase 2)

Supersedes `docs/audit/AGENT_CAPABILITY_MATRIX.md`. As of 2026-10-23, commit `f0934aa` + the later commits on
`phase2/advanced-ml-hardening`.

**Reading the table.** *Implemented* = code exists and the cited tests ran. *Verified* = checked against a hand calculation,
a reference library or a simulation (never clinical validity). *Voting* = contributes typed `HypothesisEvidence` to the
consensus. **Every agent is clinically unvalidated.** "Reliability weight" is an informal engineering default, not an
estimated quantity.

## 1. Agents in the workflow (DAG in `app/agents/registry.py`)

| Agent | Depends on | Weight | What it really does | Typed "cannot assess" outcome | Votes in consensus? | What it does NOT do | Tests |
|---|---|---|---|---|---|---|---|
| `data_intake_agent` | — | 0.9 | Validates/normalises observations, de-duplicates, scores data completeness/quality, scans free text for emergencies | (always `completed`; reports `data_quality`, `missing_data:*` warnings) | No | No device/FHIR ingestion, no unit conversion; ranges are engineering "plausible" bounds | `test_agents`, `test_input_validation` |
| `emergency_escalation_agent` | intake | 1.0 | Regex-family phrase detector with narrow negation handling, **plus** an independent deterministic pre-scan that runs before any slow agent and is checkpointed to storage | any status other than `completed` ⇒ run **fails closed** to "emergency" | No (it raises alerts, not hypotheses) | Not a trained detector; recall on real patient language **unmeasured**; English only; no structured vital-sign trigger | `test_emergency_rules` (31), `test_orchestrator*`, `test_workflow_routes` |
| `neuro_diagnostic_agent` | intake | 0.65 | Retrieval-based matching of symptom text against a 10-condition curated KB; emits per-condition `HypothesisEvidence` (diagnostic, group `symptom_text_retrieval`) | `insufficient_data` when no symptom text | **Yes** (the only voter in practice) | Not a classifier; no labelled evaluation; semantic path depends on Gemini embeddings + pgvector (unverified); not a diagnosis | `test_agents`, `test_consensus_evidence`, `test_agent_typed_statuses` |
| `knowledge_agent` | diagnostic | 0.8 | Retrieves KB entries and citation URLs for candidate conditions | — | **Context only** (its evidence type `knowledge_retrieval` is excluded from ranking because it re-reads the same text) | Citation liveness could not be checked (network); review queue exists | `test_agents`, `test_kb_integrity` |
| `signal_analysis_agent` | intake | 0.6 | Summary-metric checks (tremor amplitude, gait speed) with unvalidated 1.5×/0.8× thresholds | `skipped` (no wearable data), `abstained` (only unsupported modalities, e.g. raw EEG) | No | No raw-signal processing; consumer wearables ≠ clinical EEG/EMG | `test_agents` |
| `longitudinal_agent` | intake | 0.75 | Per-metric within-person analysis: polarity-aware change class, Student-t point test (α 0.001 signal / 0.05 note), Theil–Sen + Mann–Kendall, self-starting CUSUM/EWMA with autocorrelation gate; methods not supported by the data are `not_assessed` | `insufficient_data` when no history | No (patient-level, names no condition) | Does not diagnose; simulated false-alarm rates are 7–43 % depending on noise (MATHEMATICAL_VALIDATION.md §3); no MCID unless clinician-approved | `test_longitudinal_math` (30), `test_longitudinal_integration`, `test_metric_thresholds` |
| `medication_safety_agent` | intake | 0.85 | Brand→generic normalisation, allergy cross-check, adherence validation; interaction check through the `DDISource` interface with **seven distinct statuses**; lists drugs it could not assess | `insufficient_data` when nothing could be assessed | No | The only shipped source is a 5-rule illustrative table that can never say "no interaction" (only `combination_not_assessed`); **no licensed/open DDI database is wired up**; never proposes a dose change | `test_drug_rules` (24), `test_ddi` |
| `risk_stratification_agent` | intake, diagnostic, signal, longitudinal, emergency | 0.7 | Transparent weighted additive heuristic; "confidence" = input coverage | — | No | Weights unvalidated; a low score with missing inputs ≠ low risk (warned); not calibrated | `test_agents` |
| `personalization_agent` | risk, longitudinal, medication | 0.7 | Non-prescriptive, clinician-directed plan items | — | No | **Not** treatment support: no titration, guideline engine or recommendation | `test_agents` |
| `copilot` (`PatientCopilotAgent`) | — | 0.5 | Registered but **not scheduled** by the orchestrator | — | No | Unused | — |

## 2. Cross-cutting components

| Component | Status | Notes |
|---|---|---|
| Orchestrator (`app/orchestrator/orchestrator.py`) | Implemented | DAG with bounded concurrency (`MAX_AGENT_CONCURRENCY`), per-agent timeout = min(`AGENT_TIMEOUT_SECONDS`, time left), global `WORKFLOW_DEADLINE_SECONDS`, cancellation reaping, `TIMED_OUT` results, deterministic upstream view per agent, emergency pre-scan checkpoint, fail-closed emergency. Tested with fake slow/hung/failing agents; not load-tested. |
| Consensus (`app/core/consensus.py`) | Implemented | Uncalibrated support index over typed independent evidence; see CONSENSUS_DESIGN.md. With one voting agent it is that agent's ranking. |
| Safety gate (`app/core/safety.py`) | Implemented | Autonomy level, confidence, data quality, disagreement; restricted actions never auto-approved. |
| Persistence | Implemented (fake DB) / SQL **not executed** | Idempotency-Key → deterministic workflow id; `persist_state` marker; atomic RPC available behind `PERSIST_VIA_RPC`. |
| Notification outbox | Implemented (mock provider, smtplib stand-in) | No live delivery; default channel `none`. "executed" only after `delivered`. |
| Patient chat (`services/chat_assistant.py`) | Implemented, not validated | Deterministic emergency screen first; grounded on the latest assessment; doses/diagnoses guardrail by regex; blocked from external LLM without consent; live-model injection resistance and answer quality **untested**. |
| ML demo endpoints (`api/routes/ml_demo.py`) | Experimental, **unvalidated** | SHA-256 + restricted unpickler; OOD abstention; no real-data validation (ML_VALIDATION_RESULTS.md). Not connected to the workflow or consensus. |

## 3. Roles named in the product brief that are **not** implemented
Treatment support (needs clinical governance), a real notifier channel beyond optional SMTP, wearable/EEG/EMG/video/voice
signal processing, device ingestion, multilingual emergency detection, and any validated diagnostic model.
