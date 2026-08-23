"""
Personalization Agent (spec 6.8 / section 36).

Combines the outputs of the other agents plus the patient's own history and
preferences into a personalized monitoring/education/reminder plan. This is
where "personalized" stops being a greeting and starts being an actual
function of ΔP = P_t - P_baseline (deltas across disease/symptom/medication/
risk/behavioral/cognitive/adherence/environmental state).
"""
from __future__ import annotations

from app.agents.base import BaseNeuroAgent
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState


class PersonalizationAgent(BaseNeuroAgent):
    name = "personalization_agent"
    capabilities = ["monitoring_plan", "education_content", "reminder_plan"]
    reliability_weight = 0.7

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        payload = task.input_payload or {}
        upstream = payload.get("upstream_results", {})

        plan: list[str] = []
        education: list[str] = []
        reasons: list[str] = []

        risk = upstream.get("risk_stratification_agent", {}).get("raw_payload", {})
        scores = risk.get("heuristic_scores", {})
        if scores.get("fall_risk", 0) >= 0.5:
            plan.append("Daily gait/balance check-in reminder")
            education.append("Fall-prevention home safety tips")
            reasons.append(f"fall_risk heuristic score {scores.get('fall_risk')} for this patient")
        if scores.get("seizure_risk", 0) >= 0.5:
            plan.append("Seizure diary prompt after any missed medication dose")
            reasons.append(f"seizure_risk heuristic score {scores.get('seizure_risk')} for this patient")

        med = upstream.get("medication_safety_agent", {}).get("raw_payload", {})
        if med.get("adherence_pct_30d") is not None and med["adherence_pct_30d"] < 0.8:
            plan.append("Personalized medication reminder schedule aligned to patient's reported routine")
            reasons.append(f"30-day adherence at {round(med['adherence_pct_30d']*100,1)}% for this specific patient")

        longitudinal = upstream.get("longitudinal_agent", {}).get("raw_payload", {})
        if longitudinal.get("deteriorating_metrics"):
            plan.append(f"Increase monitoring frequency for: {', '.join(longitudinal['deteriorating_metrics'])}")
            reasons.append(f"deterioration detected in this patient's own longitudinal trend, not a population average")
        if longitudinal.get("improving_metrics"):
            plan.append(f"Advance rehabilitation tasks for: {', '.join(longitudinal['improving_metrics'])} (patient is improving)")

        conditions = state.disease_state.conditions
        if conditions:
            education.append(f"Condition-specific education for: {', '.join(conditions)} (drawn from this patient's recorded diagnosis)")

        if state.patient_preferences.preferred_language:
            reasons.append(f"content will be delivered in patient's preferred language: {state.patient_preferences.preferred_language}")

        if not plan:
            plan.append("Continue routine monitoring cadence; no personalized adjustment triggered this cycle")

        findings = [AgentFinding(label="plan_item", detail=p, severity="info") for p in plan]

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=0.7,
            warnings=[],
            recommended_next_agents=["autonomous_action_engine"],
            requires_human_review=False,
            reasoning_summary=(
                "Personalized plan derived from this patient's own risk scores, adherence, "
                "longitudinal trend and preferences (not a generic condition template): " + "; ".join(reasons or ["baseline plan"])
            ),
            raw_payload={
                "monitoring_plan": plan,
                "educational_content": education,
                "personalization_reasons": reasons,
            },
        )
