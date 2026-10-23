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
from typing import Any, Awaitable, Callable

from app.agents.registry import AGENT_DEPENDENCIES, ancestors, get_agent, all_agents
from app.core.actions import ProposedAction, build_action
from app.core.config import get_settings
from app.core.consensus import ConsensusOutcome, compute_evidence_consensus
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
    failed_agents: list[str] = field(default_factory=list)       # FAILED or TIMED_OUT (run is degraded)
    timed_out_agents: list[str] = field(default_factory=list)
    degraded: bool = False
    deadline_exceeded: bool = False
    emergency_screen: dict[str, Any] = field(default_factory=dict)


def _log(state: WorkflowState, event: str, detail: dict[str, Any]) -> None:
    state.audit_events.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "detail": detail,
    })


async def _checkpoint(state: WorkflowState, cb, name: str) -> None:
    if cb is None:
        return
    try:
        await cb(state, name)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # durability hook failure must not stop the safety workflow
        _log(state, "checkpoint_failed", {"checkpoint": name, "error": repr(exc)})


def _upstream_view(state: WorkflowState, agent_name: str) -> dict[str, Any]:
    """Results of this agent's (transitive) dependencies ONLY -- a deterministic view that does not depend on
    which unrelated agents happen to have finished first."""
    return {
        name: {
            "status": r.status, "confidence": r.confidence,
            "findings": [f.model_dump() for f in r.findings], "raw_payload": r.raw_payload,
        }
        for name, r in state.agent_results.items() if name in ancestors(agent_name)
    }


async def _run_agents(state: WorkflowState, per_agent_timeout: float, deadline_s: float, max_concurrency: int) -> None:
    """Run every registered workflow agent respecting AGENT_DEPENDENCIES, with bounded concurrency, per-agent
    timeouts and one global deadline. Every agent ends with exactly one AgentResult (completed / failed /
    timed_out / abstained / ...): nothing is dropped, and nothing is left running when this returns."""
    loop = asyncio.get_running_loop()
    deadline_at = loop.time() + deadline_s
    sem = asyncio.Semaphore(max_concurrency)
    done = {n: asyncio.Event() for n in AGENT_DEPENDENCIES}
    input_payload = state.input_payload

    def _terminal(agent_name: str, task: AgentTask, status: AgentStatus, warning: str, summary: str) -> AgentResult:
        return AgentResult(agent_name=agent_name, task_id=task.task_id, status=status, warnings=[warning],
                           requires_human_review=True, reasoning_summary=summary)

    async def run_one(agent_name: str) -> None:
        try:
            for dep in AGENT_DEPENDENCIES[agent_name]:
                await done[dep].wait()
            agent = get_agent(agent_name)
            intake = state.agent_results.get("data_intake_agent")
            payload = dict(input_payload)
            payload["observations"] = (intake.raw_payload.get("observations", [])
                                       if intake is not None else input_payload.get("symptoms", []))
            payload["upstream_results"] = _upstream_view(state, agent_name)
            task = AgentTask(workflow_id=state.workflow_id, patient_id=state.patient_id, agent_name=agent_name,
                             reason=f"stage participant for trigger '{state.trigger_reason}'", input_payload=payload)
            state.messages.append(AgentMessage(
                workflow_id=state.workflow_id, patient_id=state.patient_id, sender_agent="orchestrator",
                receiver_agent=agent_name, message_type=MessageType.TASK, priority=Priority.MEDIUM,
                payload={"reason": task.reason}))

            started = loop.time()
            async with sem:
                remaining = deadline_at - loop.time()
                if remaining <= 0:
                    state.deadline_exceeded = True
                    result = _terminal(agent_name, task, AgentStatus.TIMED_OUT, "global_deadline_exceeded_before_start",
                                       "Workflow deadline reached before this agent could start.")
                else:
                    limit = min(per_agent_timeout, remaining)
                    try:
                        result = await asyncio.wait_for(agent.execute(task, state.patient_state), timeout=limit)
                    except asyncio.TimeoutError:
                        global_hit = remaining <= per_agent_timeout
                        state.deadline_exceeded = state.deadline_exceeded or global_hit
                        result = _terminal(agent_name, task, AgentStatus.TIMED_OUT,
                                           "global_deadline_exceeded" if global_hit else f"agent_timeout:{per_agent_timeout}s",
                                           "Agent exceeded its time budget; workflow continues without its input.")
                        _log(state, "agent_timed_out", {"agent": agent_name, "limit_s": round(limit, 3), "global": global_hit})
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # one agent must never crash the workflow
                        result = _terminal(agent_name, task, AgentStatus.FAILED, f"agent_exception:{exc!r}",
                                           "Agent raised an exception; degraded mode -- workflow continues without this agent's input.")
                        _log(state, "agent_failed", {"agent": agent_name, "error": repr(exc)})
            result.duration_ms = round((loop.time() - started) * 1000, 1)

            status = str(result.status)
            if status == "timed_out":
                state.timed_out_agents.append(agent_name)
            if status in ("failed", "timed_out") and agent_name not in state.failed_agents:
                state.failed_agents.append(agent_name)
            state.agent_results[agent_name] = result
            state.messages.append(AgentMessage(
                workflow_id=state.workflow_id, patient_id=state.patient_id, sender_agent=agent_name,
                receiver_agent="orchestrator", message_type=MessageType.ANALYSIS, priority=Priority.MEDIUM,
                payload={"status": result.status, "reasoning_summary": result.reasoning_summary},
                confidence=result.confidence, evidence=result.evidence))
            _log(state, "agent_completed", {"agent": agent_name, "status": result.status,
                                            "confidence": result.confidence, "duration_ms": result.duration_ms})
        finally:
            done[agent_name].set()   # release dependents even on cancellation/crash so nothing waits forever

    tasks = [asyncio.create_task(run_one(n), name=f"agent:{n}") for n in AGENT_DEPENDENCIES]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)   # reap: no orphan tasks after cancellation
        raise
    # deterministic ordering of the results mapping (completion order is not)
    state.agent_results = {n: state.agent_results[n] for n in AGENT_DEPENDENCIES if n in state.agent_results}


async def run_workflow(
    *,
    patient_id: str,
    trigger_reason: str,
    input_payload: dict[str, Any],
    patient_state: PatientState,
    workflow_id: str | None = None,
    on_checkpoint: Callable[[WorkflowState, str], Awaitable[None]] | None = None,
) -> WorkflowState:
    """`workflow_id` lets a caller make the run idempotent (same id -> same persisted row).
    `on_checkpoint(state, name)` is awaited after the emergency pre-scan so the caller can durably record an
    emergency BEFORE the slow agents run; a failing checkpoint is logged and never blocks the workflow."""
    workflow_id = workflow_id or str(uuid.uuid4())
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
    await _checkpoint(state, on_checkpoint, "prescan")

    settings = get_settings()
    await _run_agents(state, settings.agent_timeout_seconds, settings.workflow_deadline_seconds,
                      max(1, int(settings.max_agent_concurrency)))

    emergency_result = state.agent_results.get("emergency_escalation_agent")
    if emergency_result is not None:
        # OR with the independent pre-scan; an emergency agent that did not COMPLETE (failed, timed out, abstained)
        # can never clear the flag: fail closed.
        state.is_emergency = state.is_emergency or bool(emergency_result.raw_payload.get("is_emergency"))
        if str(emergency_result.status) != "completed":
            state.is_emergency = True
            emergency_result.raw_payload.setdefault("is_emergency", True)
            emergency_result.raw_payload["failed_closed"] = True
            _log(state, "emergency_agent_failed_closed", {"agent": emergency_result.agent_name,
                                                          "status": str(emergency_result.status)})
        if state.is_emergency:
            state.messages.append(AgentMessage(
                workflow_id=workflow_id, patient_id=patient_id,
                sender_agent=emergency_result.agent_name, receiver_agent="orchestrator",
                message_type=MessageType.ESCALATION, priority=Priority.CRITICAL,
                payload=emergency_result.raw_payload, confidence=1.0,
                requires_response=True,
            ))
            _log(state, "emergency_detected", emergency_result.raw_payload)

    # --- Consensus over TYPED, hypothesis-specific evidence ---------------------------------------
    # Only agents that attach HypothesisEvidence take part; patient-level agents (signal, longitudinal, risk,
    # medication) abstain from naming a disease, and the knowledge agent's lookups are context only. The result is
    # an UNCALIBRATED ordering, never a probability (docs/phase2/CONSENSUS_DESIGN.md).
    diagnostic = state.agent_results.get("neuro_diagnostic_agent")
    intake = state.agent_results.get("data_intake_agent")
    data_quality_q = intake.raw_payload.get("data_quality") if intake is not None and str(intake.status) == "completed" else None
    if any(r.hypothesis_evidence for r in state.agent_results.values()):
        reliability_weights = {name: a.reliability_weight for name, a in all_agents().items()}
        state.consensus = compute_evidence_consensus(
            results=list(state.agent_results.values()),
            reliability_weights=reliability_weights,
            data_quality=data_quality_q,
        )
        _log(state, "consensus_computed", {
            "top_hypothesis": state.consensus.top_hypothesis,
            "disagreement": state.consensus.disagreement,
            "action_permitted": state.consensus.action_permitted,
            "score_kind": state.consensus.score_kind, "ties": state.consensus.ties,
        })
        state.messages.append(AgentMessage(
            workflow_id=workflow_id, patient_id=patient_id,
            sender_agent="orchestrator", receiver_agent="orchestrator",
            message_type=MessageType.CONSENSUS_RESULT, priority=Priority.MEDIUM,
            payload={
                "top_hypothesis": state.consensus.top_hypothesis,
                "disagreement": state.consensus.disagreement,
                "score_kind": state.consensus.score_kind, "calibrated": False,
                "hypotheses": [h.__dict__ for h in state.consensus.hypotheses],
            },
            confidence=1 - state.consensus.disagreement,
        ))

    # --- Personalization -> Action proposals ---------------------------
    personalization = state.agent_results.get("personalization_agent")
    data_quality = data_quality_q if data_quality_q is not None else 0.5
    # No unique supported leader (tie / contradicted / unsupported / no evidence) -> zero confidence, so the safety
    # gate blocks autonomy rather than inheriting a number from an unrelated hypothesis.
    top_confidence = (next((h.consensus_score for h in state.consensus.hypotheses
                            if h.hypothesis == state.consensus.top_hypothesis), 0.0)
                      if state.consensus else (diagnostic.confidence if diagnostic and str(diagnostic.status) == "completed" else 0.0))
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
            evidence=(state.agent_results["emergency_escalation_agent"].raw_payload.get("matched_phrases", [])
                      if "emergency_escalation_agent" in state.agent_results else state.emergency_screen.get("matched", [])),
            is_emergency=True,
            disagreement=disagreement,
            top_confidence=top_confidence,
            data_quality=data_quality,
            clinician_restricted_actions=patient_state.clinician_constraints.restricted_actions,
        )
        state.proposed_actions.insert(0, emergency_action)
        _log(state, "emergency_action_proposed", {"action_id": emergency_action.action_id})

    state.degraded = bool(state.failed_agents) or state.deadline_exceeded
    state.status = "completed"  # kept for DB/API compatibility; use `degraded` / `failed_agents` for run health
    state.completed_at = datetime.now(timezone.utc).isoformat()
    _log(state, "workflow_completed", {"num_actions": len(state.proposed_actions),
                                      "degraded": state.degraded, "failed_agents": state.failed_agents,
                                      "timed_out_agents": state.timed_out_agents,
                                      "deadline_exceeded": state.deadline_exceeded})
    return state
