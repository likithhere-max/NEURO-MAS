"""
Autonomous Emergency / Escalation Agent (spec 6.11).

Architecture mandated by the spec:
    Sensor/Event -> Deterministic Safety Detector -> Emergency Agent
    -> Orchestrator -> Safety Policy Engine -> Action

This agent does NOT itself decide to escalate -- it packages the
deterministic detector's output (from app.services.clinical_rules) into a
structured finding for the orchestrator/safety engine to act on. The actual
policy decision ("is escalation permitted/required") lives in
app.core.safety, deliberately kept out of any single agent so the emergency
path is never dependent on a model call succeeding.
"""
from __future__ import annotations

from app.agents.base import BaseNeuroAgent
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.clinical_rules import detect_emergency_detailed


def collect_texts(payload: dict) -> list[str]:
    """Every free-text-bearing field the patient or caregiver supplied -- from the
    normalised observations AND the raw input (so a data-intake failure can never
    blind the emergency scan). Audit fix: previously only `free_text or name` was
    scanned per observation, so a benign free_text hid an alarming name."""
    texts: list[str] = []
    def add(v):
        if isinstance(v, str) and v.strip():
            texts.append(v)
    for key in ("observations", "symptoms"):
        for o in (payload.get(key) or []):
            if isinstance(o, dict):
                for f in ("name", "free_text", "note", "severity", "onset"):
                    add(o.get(f))
            else:
                add(o if isinstance(o, str) else None)
    add(payload.get("caregiver_notes"))
    for ml in (payload.get("medication_logs") or []):
        if isinstance(ml, dict):
            add(ml.get("note"))
    return texts


class EmergencyEscalationAgent(BaseNeuroAgent):
    name = "emergency_escalation_agent"
    capabilities = ["deterministic_emergency_detection", "escalation_packaging"]
    reliability_weight = 1.0  # deterministic safety-critical path; always fully trusted

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        payload = task.input_payload or {}
        detail = detect_emergency_detailed(collect_texts(payload))
        is_emergency, matched = detail["is_emergency"], detail["matched"]

        findings = []
        if is_emergency:
            findings.append(AgentFinding(
                label="emergency_detected",
                detail=f"Deterministic detector matched: {', '.join(matched)}",
                severity="critical",
            ))

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=1.0 if is_emergency else 0.0,  # rule fired / not fired -- NOT a probability of emergency
            warnings=[],
            recommended_next_agents=[],
            requires_human_review=is_emergency,
            reasoning_summary=(
                "Deterministic phrase-matching emergency detector "
                + ("MATCHED known emergency indicators." if is_emergency else "found no emergency indicators.")
            ),
            raw_payload={"is_emergency": is_emergency, "matched_phrases": matched,
                         "families": detail["families"], "negated_matches": detail["negated"]},
        )
