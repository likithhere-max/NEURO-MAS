"""
Orchestrator / Conductor Agent (spec section 6.1 / 34).

Runs the closed loop described in the spec:

    DATA INTAKE -> ROUTE TASKS -> PARALLEL AGENTS -> COLLECT RESULTS
    -> CONFLICT DETECTION -> CONSENSUS -> SAFETY ENGINE -> PERSONALIZATION
    -> ACTION -> AUDIT

Implemented as an explicit async state machine over WORKFLOW_STAGES rather
than a LangGraph graph: for a project this size, an explicit function with a
typed WorkflowState produces the same auditable, inspectable state trail
LangGraph would give us, without adding a framework dependency the project
doesn't otherwise need (see docs/ARCHITECTURE.md, "why not LangGraph").
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.agents.registry import WORKFLOW_STAGES, all_agents, get_agent
from app.core.actions import ProposedAction, build_action
from app.core.config import get_settings
from app.core.consensus import ConsensusOutcome, compute_consensus
from app.core.messaging import AgentMessage, MessageType, Priority
from app.schemas.agent import AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.clinical_rules import detect_emergency_detailed


@dataclass
class WorkflowState:
    workflow_id: str
    patient_id: str
    trigger_reason: str
    input_payload: dict[str, Any]
    patient_state: PatientState
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    messages: list[AgentMessage] = field(default_factory=list)
    audit_events: list[dict[str, Any]] = field(default_factory=list)
    consensus: ConsensusOutcome | None = None
    is_emergency: bool = False
    proposed_actions: list[ProposedAction] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    status: str = "running"
    # Audit fix: a run in which agents failed is no longer indistinguishable from a clean run.
    failed_agents: list[str] = field(default_factory=list)
    degraded: bool = False
    emergency_screen: dict[str, Any] = field(default_factory=dict)


def _log(state: WorkflowState, event: str, detail: dict[str, Any]) -> None:
    state.audit_events.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "detail": detail,
    })


async def run_workflow(
    *,
    patient_id: str,
    trigger_reason: str,
    input_payload: dict[str, Any],
    patient_state: PatientState,
) -> WorkflowState:
    workflow_id = str(uuid.uuid4())
    state = WorkflowState(
        workflow_id=workflow_id,
        patient_id=patient_id,
        trigger_reason=trigger_reason,
        input_payload=input_payload,
        patient_state=patient_state,
    )
    _log(state, "workflow_started", {"trigger_reason": trigger_reason})

    # Independent deterministic emergency pre-scan on the RAW input. It runs
    # before any agent/LLM call and does not depend on data_intake or the
    # emergency agent succeeding. Fail-CLOSED: if the scan itself errors we
    # treat the case as a possible emergency (over-triage beats a missed stroke).
    from app.agents.emergency import collect_texts
    try:
        prescan = detect_emergency_detailed(collect_texts(input_payload or {}))
        state.emergency_screen = {"source": "prescan", **prescan}
        if prescan["is_emergency"]:
            state.is_emergency = True
    except Exception as exc:
        state.emergency_screen = {"source": "prescan", "is_emergency": True, "matched": [], "error": repr(exc),
                                  "note": "emergency screen crashed -> fail-closed"}
        state.is_emergency = True
    _log(state, "emergency_prescan", {k: v for k, v in state.emergency_screen.items() if k != "negated"})

    timeout_s = get_settings().agent_timeout_seconds

    for stage_agents in WORKFLOW_STAGES:
        for agent_name in stage_agents:
            agent = get_agent(agent_name)

            # Build this agent's task payload: its own input plus every
            # upstream result so far (this is the "shared blackboard").
            task_payload = dict(input_payload)
            task_payload["observations"] = state.agent_results.get(
                "data_intake_agent"
            ).raw_payload.get("observations", []) if "data_intake_agent" in state.agent_results else input_payload.get("symptoms", [])
            task_payload["upstream_results"] = {
                name: {
                    "status": r.status,
                    "confidence": r.confidence,
                    "findings": [f.model_dump() for f in r.findings],
                    "raw_payload": r.raw_payload,
                }
                for name, r in state.agent_results.items()
            }

            task = AgentTask(
                workflow_id=workflow_id,
                patient_id=patient_id,
                agent_name=agent_name,
                reason=f"stage participant for trigger '{trigger_reason}'",
                input_payload=task_payload,
            )

            state.messages.append(AgentMessage(
                workflow_id=workflow_id, patient_id=patient_id,
                sender_agent="orchestrator", receiver_agent=agent_name,
                message_type=MessageType.TASK, priority=Priority.MEDIUM,
                payload={"reason": task.reason},
            ))

            try:
                result = await asyncio.wait_for(agent.execute(task, patient_state), timeout=timeout_s)
            except Exception as exc:  # never let one agent crash the workflow (incl. TimeoutError)
                state.failed_agents.append(agent_name)
                result = AgentResult(
                    agent_name=agent_name,
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    warnings=[f"agent_exception:{exc!r}"],
                    requires_human_review=True,
                    reasoning_summary="Agent raised an exception; degraded mode -- workflow continues without this agent's input.",
                )
                _log(state, "agent_failed", {"agent": agent_name, "error": repr(exc)})

            if result.status == AgentStatus.FAILED and agent_name not in state.failed_agents:
                state.failed_agents.append(agent_name)
            state.agent_results[agent_name] = result
            state.messages.append(AgentMessage(
                workflow_id=workflow_id, patient_id=patient_id,
                sender_agent=agent_name, receiver_agent="orchestrator",
                message_type=MessageType.ANALYSIS, priority=Priority.MEDIUM,
                payload={"status": result.status, "reasoning_summary": result.reasoning_summary},
                confidence=result.confidence,
                evidence=result.evidence,
            ))
            _log(state, "agent_completed", {
                "agent": agent_name, "status": result.status, "confidence": result.confidence,
            })

            if agent_name == "emergency_escalation_agent":
                # OR with the independent pre-scan; a FAILED emergency agent can never clear the flag.
                state.is_emergency = state.is_emergency or bool(result.raw_payload.get("is_emergency"))
                if result.status == AgentStatus.FAILED:
                    state.is_emergency = True  # fail-closed: the safety-critical agent did not complete
                    result.raw_payload.setdefault("is_emergency", True)
                    result.raw_payload["failed_closed"] = True
                    _log(state, "emergency_agent_failed_closed", {"agent": agent_name})
                if state.is_emergency:
                    state.messages.append(AgentMessage(
                        workflow_id=workflow_id, patient_id=patient_id,
                        sender_agent=agent_name, receiver_agent="orchestrator",
                        message_type=MessageType.ESCALATION, priority=Priority.CRITICAL,
                        payload=result.raw_payload, confidence=1.0,
                        requires_response=True,
                    ))
                    _log(state, "emergency_detected", result.raw_payload)

    # --- Conflict detection + consensus -------------------------------
    diagnostic = state.agent_results.get("neuro_diagnostic_agent")
    candidates = (diagnostic.raw_payload.get("candidate_conditions", []) if diagnostic else [])
    # Only the diagnostic agent produces a hypothesis-specific score (retrieval similarity of
    # THAT candidate). Signal/longitudinal/risk produce patient-level findings that are not tied
    # to a named disease, so they ABSTAIN rather than vote with an unrelated global confidence.
    # The knowledge agent only looks up the same candidates (circular), so it does not vote either.
    hypotheses = {c["preferred_name"]: ["neuro_diagnostic_agent"] for c in candidates[:3]}
    hypothesis_support = {
        c["preferred_name"]: {"neuro_diagnostic_agent": float(c.get("confidence", 0.0))} for c in candidates[:3]
    }

    conflicts = []
    if diagnostic and diagnostic.raw_payload.get("red_flags"):
        conflicts.append(f"diagnostic red flags present: {diagnostic.raw_payload['red_flags']}")

    if hypotheses:
        reliability_weights = {name: get_agent(name).reliability_weight for name in all_agents()}
        state.consensus = compute_consensus(
            results=list(state.agent_results.values()),
            reliability_weights=reliability_weights,
            data_quality=state.agent_results.get("data_intake_agent").raw_payload.get("data_quality", 0.5)
            if "data_intake_agent" in state.agent_results else 0.5,
            hypotheses=hypotheses,
            hypothesis_support=hypothesis_support,
        )
        _log(state, "consensus_computed", {
            "top_hypothesis": state.consensus.top_hypothesis,
            "disagreement": state.consensus.disagreement,
            "action_permitted": state.consensus.action_permitted,
        })
        state.messages.append(AgentMessage(
            workflow_id=workflow_id, patient_id=patient_id,
            sender_agent="orchestrator", receiver_agent="orchestrator",
            message_type=MessageType.CONSENSUS_RESULT, priority=Priority.MEDIUM,
            payload={
                "top_hypothesis": state.consensus.top_hypothesis,
                "disagreement": state.consensus.disagreement,
                "hypotheses": [h.__dict__ for h in state.consensus.hypotheses],
            },
            confidence=1 - state.consensus.disagreement,
        ))

    # --- Personalization -> Action proposals ---------------------------
    personalization = state.agent_results.get("personalization_agent")
    data_quality = state.agent_results.get("data_intake_agent").raw_payload.get("data_quality", 0.5) if "data_intake_agent" in state.agent_results else 0.5
    top_confidence = state.consensus.hypotheses[0].consensus_score if state.consensus and state.consensus.hypotheses else (diagnostic.confidence if diagnostic else 0.0)
    disagreement = state.consensus.disagreement if state.consensus else 0.0

    if personalization:
        plan_items = personalization.raw_payload.get("monitoring_plan", [])
        reasons = personalization.raw_payload.get("personalization_reasons", [])
        for item in plan_items:
            action = build_action(
                patient_id=patient_id,
                workflow_id=workflow_id,
                plan_item=item,
                reason=f"{item} (basis: {'; '.join(reasons) if reasons else 'routine personalized monitoring'})",
                evidence=[m.description for r in state.agent_results.values() for m in r.evidence][:5],
                is_emergency=state.is_emergency,
                disagreement=disagreement,
                top_confidence=top_confidence,
                data_quality=data_quality,
                clinician_restricted_actions=patient_state.clinician_constraints.restricted_actions,
            )
            state.proposed_actions.append(action)
            _log(state, "action_proposed", {"action_type": action.action_type, "status": action.status})

    if state.is_emergency:
        emergency_action = build_action(
            patient_id=patient_id,
            workflow_id=workflow_id,
            plan_item="emergency escalation",
            action_type="emergency_escalation",
            reason="Deterministic emergency detector matched a known red-flag phrase.",
            evidence=state.agent_results["emergency_escalation_agent"].raw_payload.get("matched_phrases", []),
            is_emergency=True,
            disagreement=disagreement,
            top_confidence=top_confidence,
            data_quality=data_quality,
            clinician_restricted_actions=patient_state.clinician_constraints.restricted_actions,
        )
        state.proposed_actions.insert(0, emergency_action)
        _log(state, "emergency_action_proposed", {"action_id": emergency_action.action_id})

    state.degraded = bool(state.failed_agents)
    state.status = "completed"  # kept for DB/API compatibility; use `degraded` / `failed_agents` for run health
    state.completed_at = datetime.now(timezone.utc).isoformat()
    _log(state, "workflow_completed", {"num_actions": len(state.proposed_actions),
                                      "degraded": state.degraded, "failed_agents": state.failed_agents})
    return state
