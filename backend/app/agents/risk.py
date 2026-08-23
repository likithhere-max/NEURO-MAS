"""
Risk Stratification Agent (spec 6.6).

Computes disease-relevant risk categories from an explicit, inspectable
weighted-factor heuristic. Every score is labeled `heuristic_score` --
NEVER `validated_clinical_risk_score` -- because none of these formulas have
undergone clinical validation. That distinction is enforced in the output
schema itself, not just in prose, so downstream consumers (UI, orchestrator)
can't accidentally present a heuristic as validated.
"""
from __future__ import annotations

from app.agents.base import BaseNeuroAgent
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState


def _bucket(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


class RiskStratificationAgent(BaseNeuroAgent):
    name = "risk_stratification_agent"
    capabilities = ["fall_risk", "seizure_risk", "deterioration_risk", "medication_risk"]
    reliability_weight = 0.7

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        payload = task.input_payload or {}
        upstream = payload.get("upstream_results", {})  # results from earlier agents in this workflow

        factors: dict[str, float] = {}

        # Fall risk: gait slowing + tremor increase + longitudinal deterioration
        fall_score = 0.0
        signal = upstream.get("signal_analysis_agent", {})
        if any(f.get("label") == "gait_slowing" for f in signal.get("findings", [])):
            fall_score += 0.4
        if any(f.get("label") == "tremor_increase" for f in signal.get("findings", [])):
            fall_score += 0.25
        longitudinal = upstream.get("longitudinal_agent", {})
        if longitudinal.get("raw_payload", {}).get("deteriorating_metrics"):
            fall_score += 0.25
        if state.adherence_state.missed_checkins_7d and state.adherence_state.missed_checkins_7d >= 3:
            fall_score += 0.1
        fall_score = min(fall_score, 1.0)

        # Seizure risk: adherence + prior seizure history + intake urgency
        seizure_score = 0.0
        if state.medication_state.adherence_pct_30d is not None and state.medication_state.adherence_pct_30d < 0.8:
            seizure_score += 0.35
        if state.seizure_history:
            recent = [e for e in state.seizure_history if e.get("recent")]
            seizure_score += min(0.1 * len(recent), 0.4)
        intake = upstream.get("data_intake_agent", {})
        if intake.get("raw_payload", {}).get("urgency") in {"high", "critical"}:
            seizure_score += 0.3
        seizure_score = min(seizure_score, 1.0)

        # General deterioration risk: blend of longitudinal + diagnostic confidence
        deterioration_score = 0.0
        if longitudinal.get("raw_payload", {}).get("deteriorating_metrics"):
            deterioration_score += 0.5
        diagnostic = upstream.get("neuro_diagnostic_agent", {})
        top_candidates = diagnostic.get("raw_payload", {}).get("candidate_conditions", [])
        if top_candidates and top_candidates[0].get("confidence", 0) > 0.5:
            deterioration_score += 0.25
        deterioration_score = min(deterioration_score, 1.0)

        factors = {
            "fall_risk": round(fall_score, 2),
            "seizure_risk": round(seizure_score, 2),
            "deterioration_risk": round(deterioration_score, 2),
        }

        findings = [
            AgentFinding(
                label=f"heuristic_score:{name}",
                detail=f"{_bucket(score)} ({score})",
                severity=_bucket(score) if _bucket(score) != "low" else "info",
                value=score,
            )
            for name, score in factors.items()
        ]

        overall = max(factors.values()) if factors else 0.0
        confidence = 0.65 if any(factors.values()) else 0.3

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=confidence,
            warnings=["heuristic_score_not_a_validated_clinical_risk_score"],
            recommended_next_agents=["personalization_agent"],
            requires_human_review=overall >= 0.5,
            reasoning_summary=(
                f"Computed heuristic risk scores from available signal/longitudinal/diagnostic "
                f"inputs. Overall bucket: {_bucket(overall)}."
            ),
            raw_payload={
                "heuristic_scores": factors,
                "overall_bucket": _bucket(overall),
                "score_type": "heuristic_score",  # explicitly not "validated_clinical_risk_score"
            },
        )
