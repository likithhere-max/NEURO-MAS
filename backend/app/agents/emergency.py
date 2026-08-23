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
from app.services.clinical_rules import detect_emergency


class EmergencyEscalationAgent(BaseNeuroAgent):
    name = "emergency_escalation_agent"
    capabilities = ["deterministic_emergency_detection", "escalation_packaging"]
    reliability_weight = 1.0  # deterministic safety-critical path; always fully trusted

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        payload = task.input_payload or {}
        observations = payload.get("observations", [])
        texts = [o.get("free_text", "") or o.get("name", "") or o.get("note", "") for o in observations]

        is_emergency, matched = detect_emergency(texts)

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
            confidence=1.0 if is_emergency else 0.0,
            warnings=[],
            recommended_next_agents=[],
            requires_human_review=is_emergency,
            reasoning_summary=(
                "Deterministic phrase-matching emergency detector "
                + ("MATCHED known emergency indicators." if is_emergency else "found no emergency indicators.")
            ),
            raw_payload={"is_emergency": is_emergency, "matched_phrases": matched},
        )
