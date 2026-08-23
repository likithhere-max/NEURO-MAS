# NEURO-MAS Architecture

**A Multi-Agent Collaborative System for Autonomous and Personalized Neurological Healthcare**

This document is the target architecture for the build produced in this session. It exists so that
every design decision can be traced back to one of the project's three pillars: **multi-agent
collaboration**, **autonomous operation**, and **personalized neurological healthcare**. If a
component in this codebase can't be explained by at least one of those three, it shouldn't be here.

## 1. Why this codebase looks the way it does

No prior codebase was uploaded to this session (see the project's opening exchange) — only a
neurological disease knowledge-base seed pack (`backend/data/`). This is therefore a from-scratch
build of a **runnable skeleton**, not an audit/refactor of an existing system. Section 44 of the
original brief ("audit before rewriting") doesn't apply for that reason; instead this doc plays the
role the target-architecture section of that audit would have played.

Given the brief's own instruction not to over-engineer a student project (its section 29), this
build prioritizes: a working closed loop end-to-end, genuine multi-agent collaboration you can see
happen, and honesty about what's simulated. It deliberately does not implement everything in the
original 44-section brief in one pass — see `docs/ROADMAP.md` for what's deferred and why.

## 2. System diagram

```text
                         PATIENT / CAREGIVER / CLINICIAN
                                      |
                         Symptoms, medication logs, wearable
                         summaries, vitals, caregiver notes
                                      v
                         ┌─────────────────────────────┐
                         │      DATA INTAKE AGENT       │  <- normalizes, flags missing data,
                         └──────────────┬──────────────┘     deterministic urgency triage
                                        v
                         ┌─────────────────────────────┐
                         │  EMERGENCY ESCALATION AGENT │  <- deterministic red-flag phrase match
                         └──────────────┬──────────────┘     (never an LLM judgment call)
                                        v
              ┌─────────────────────────┼────────────────────────┐
              v                         v                        v
     ┌────────────────┐      ┌───────────────────┐     ┌────────────────────┐
     │  NEURO-         │      │  SIGNAL ANALYSIS   │     │                     │
     │  DIAGNOSTIC     │      │  AGENT             │     │                     │
     │  AGENT          │      │  (tremor/gait       │     │                     │
     │  (symptom-KB     │      │  threshold checks;  │     │                     │
     │  overlap)        │      │  MODEL_NOT_AVAILABLE│     │                     │
     └────────┬────────┘      │  for EEG/EMG/video) │     │                     │
              │                └──────────┬─────────┘     │                     │
              └─────────────┬─────────────┘               │                     │
                             v                             │                     │
              ┌────────────────────────────┐               │                     │
              │  LONGITUDINAL AGENT        │  <- baseline vs current trend       │
              │  MEDICATION SAFETY AGENT   │  <- interactions, allergies,        │
              │  KNOWLEDGE / RAG AGENT     │     adherence; source citations     │
              └──────────────┬─────────────┘               │                     │
                             v                                                    │
              ┌────────────────────────────┐                                     │
              │  RISK STRATIFICATION AGENT │  <- heuristic_score, never claimed  │
              └──────────────┬─────────────┘     "validated_clinical_risk_score" │
                             v
              ┌────────────────────────────┐
              │  PERSONALIZATION AGENT     │  <- ΔP vs this patient's own baseline
              └──────────────┬─────────────┘
                             v
                 ┌───────────────────────┐
                 │  CONSENSUS ENGINE     │  C(H) = Σ(w·q·e·s) / Σ(w·q);  D = 1 - max C(H)
                 └──────────┬────────────┘
                             v
                 ┌───────────────────────┐
                 │  SAFETY POLICY ENGINE │  <- deterministic, catalog-driven, the ONLY
                 └──────────┬────────────┘     authority on autonomy levels 0-5
                             v
                 ┌───────────────────────┐
                 │  AUTONOMOUS ACTION    │  <- proposes/executes low-risk actions;
                 │  ENGINE               │     restricted actions ALWAYS need a clinician
                 └──────────┬────────────┘
                             v
        PATIENT UI      CAREGIVER ALERT      CLINICIAN DASHBOARD
                             |
                             v
                    AUDIT LOG (every step, every agent, every decision)
```

## 3. Why an explicit async state machine instead of LangGraph

Section 34 of the brief allows substituting "another orchestration framework" if it's already better
integrated, and explicitly says not to add frameworks the project doesn't need (section 33). This
build implements `WorkflowState` (`backend/app/orchestrator/orchestrator.py`) as a plain dataclass
threaded through an async function over `WORKFLOW_STAGES` groups. It produces the same properties a
LangGraph `StateGraph` would give here — typed state, an inspectable step-by-step trail, resumability
in principle — without adding a graph-execution dependency to a project whose actual complexity is in
the *agents and safety logic*, not the control flow between them. If the project later needs
conditional branching, cyclic re-analysis requests, or multi-turn human-in-the-loop pauses baked into
the graph itself, LangGraph becomes worth its weight — that's listed in `docs/ROADMAP.md`.

## 4. Agent responsibilities (implemented)

| Agent | File | Reliability weight | What it actually does |
|---|---|---|---|
| Data Intake | `agents/data_intake.py` | 0.90 | Normalizes raw input, computes `data_quality`, deterministic urgency triage |
| Emergency Escalation | `agents/emergency.py` | 1.00 | Deterministic red-flag phrase match only — no model in the safety-critical path |
| Neuro-Diagnostic | `agents/diagnostic.py` | 0.65 | Symptom-term overlap against the curated KB; confidence capped at 0.75 |
| Signal Analysis | `agents/signal_analysis.py` | 0.60 | Threshold checks on pre-extracted wearable summaries; `MODEL_NOT_AVAILABLE` for EEG/EMG/video/voice |
| Longitudinal / Biomarker | `agents/longitudinal.py` | 0.75 | Baseline-vs-current percent-change trend detection |
| Medication Safety | `agents/medication.py` | 0.85 | Curated interaction/allergy table lookup; **never** proposes a dose |
| Knowledge / RAG | `agents/knowledge.py` | 0.80 | Lexical retrieval over the curated, source-tagged disease KB |
| Risk Stratification | `agents/risk.py` | 0.70 | Weighted-factor `heuristic_score` (fall/seizure/deterioration) |
| Personalization | `agents/personalization.py` | 0.70 | ΔP-driven monitoring/education plan, tied to this patient's own data |
| Patient Copilot | `agents/copilot.py` | 0.50 | Plain-language explanation of an already-computed decision only |

The **Orchestrator** (`orchestrator/orchestrator.py`) is not itself an agent in the registry — it's
the conductor that routes tasks, aggregates results into `upstream_results` for downstream agents (the
shared blackboard), runs consensus, and hands off to the safety engine.

## 5. Consensus and disagreement

Implements the spec formula directly (`core/consensus.py`):

```
C(H) = Σ(w_i · q_i · e_i · s_i) / Σ(w_i · q_i)
D = 1 - max_H C(H)
```

`w_i` = agent reliability weight, `q_i` = data quality (falls back to the agent's own confidence when
unavailable), `e_i` = mean evidence strength, `s_i` = the agent's own confidence as its "support" for
the hypothesis. This is explicitly **not** claimed to be clinically validated — it's a configurable,
auditable aggregation, per the brief's own caveat. The threshold (`consensus_disagreement_threshold`,
default 0.4) is an environment-configurable value in `core/config.py`.

## 6. Safety architecture

`core/safety.py` is the single choke point every proposed action passes through — no agent, and no
LLM output, can bypass it. Rules are plain Python if/else (see `evaluate_action`), so the entire
decision space is unit-testable (`backend/tests/test_safety.py` exhaustively covers it):

1. Restricted action types (`prescription_modification`, `medication_initiation`,
   `diagnosis_confirmation`, `invasive_intervention`) **always** require clinician authorization —
   no confidence score, consensus, or emergency flag overrides this.
2. Clinician-defined per-patient restrictions override everything else.
3. The dedicated `emergency_escalation` action type executes immediately, independent of
   consensus/disagreement — but critically, this is scoped to that one action type. An active
   emergency does **not** silently reclassify unrelated low-risk actions (a medication reminder)
   as an escalation; those still pass their own confidence/disagreement gate. (This was a real bug
   caught and fixed during this session's own integration testing — see git history / test
   `test_active_emergency_does_not_reclassify_unrelated_low_risk_action`.)
4. Low-risk actions need confidence, data quality, and disagreement all within configured bounds.
5. Notification-only (RECOMMEND-level) actions are permitted more liberally since they only inform
   a human, never act on the patient directly.

## 7. Data model

`backend/app/schemas/patient_state.py` implements the P_t = [D_t, S_t, M_t, R_t, B_t, C_t, A_t, E_t]
vector from section 36 as named, inspectable sub-objects (not an opaque float array), persisted via
SQLAlchemy models in `backend/app/db/models.py` (`Patient`, `Condition`, `Observation`, `Medication`,
`Allergy`, `LongitudinalMetric`, `Workflow`, `AgentEventLog`, `Alert`, `ActionRecord`, `AuditEvent`).
Default database is SQLite (zero-setup for a student project); switching `DATABASE_URL` to Postgres
is a config change, not a rewrite.

## 8. Title-lock check for this build

- **Multi-agent**: 10 agents with typed messages (`core/messaging.py`), a shared blackboard
  (`upstream_results` passed stage-to-stage), a consensus engine that can detect disagreement between
  agents, and a UI that visibly separates each agent's status/confidence/reasoning.
- **Autonomous**: explicit `AutonomyLevel` enum (0–5), an action catalog that maps every action type
  to a level, and a safety engine that is the sole authority on what executes without a human.
- **Personalized**: the Personalization Agent's output is provably a function of *this patient's*
  adherence/risk/longitudinal data (see `personalization_reasons` in its output), not a
  condition-name template.
- **Neurological-only**: the knowledge base (`backend/data/diseases_curated.json`) covers only
  neurological conditions (epilepsy, Parkinson's, Alzheimer's, stroke, peripheral neuropathy, MS,
  ALS, Huntington's, myasthenia gravis, Guillain-Barré); no other specialty was introduced anywhere
  in the schema, agents, or UI.
