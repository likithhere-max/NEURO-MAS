"""
Risk Stratification Agent (spec 6.6).

Computes disease-relevant risk categories from an explicit, inspectable
weighted-factor heuristic. Every score is labeled `heuristic_score` --
NEVER `validated_clinical_risk_score` -- because none of these formulas have
undergone clinical validation. That distinction is enforced in the output
schema itself, not just in prose, so downstream consumers (UI, orchestrator)
can't accidentally present a heuristic as validated.

Gemini's role here is deliberately narrow: it only turns the already-computed
deterministic scores into a plain-language explanation. It never computes,
adjusts, or invents a risk number -- that stays 100% heuristic/auditable, per
the architecture decision that safety-relevant scoring must stay deterministic.
"""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from app.agents.base import BaseNeuroAgent
from app.ai.gemini_service import GeminiServiceError, generate_structured
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState

logger = logging.getLogger("neuromas.agents.risk")


def _bucket(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


class _RiskNarrative(BaseModel):
    """Gemini's structured output -- ONLY a narrative explanation of numbers
    we already computed. This schema has no numeric fields on purpose, so
    Gemini has no field to sneak a fabricated score into."""
    explanation: str


async def _explain_risk_with_gemini(factors: dict[str, float], overall_bucket: str) -> str | None:
    """
    Best-effort: turns the already-computed, deterministic heuristic scores
    into a clinician-readable explanation. Returns None on any failure so the
    caller falls back to the plain template sentence -- this must never
    crash the agent or cause it to invent risk numbers.
    """
    factor_lines = "\n".join(
        f"- {name.replace('_', ' ')}: {score} ({_bucket(score)})" for name, score in factors.items()
    )
    prompt = (
        f"These neurological risk scores were computed by a deterministic heuristic, "
        f"not by you. Overall bucket: {overall_bucket}.\n{factor_lines}\n\n"
        "Write a 2-3 sentence plain-language explanation of what these scores mean "
        "for a clinician reviewing this patient. Do not invent new numbers, symptoms, "
        "or diagnoses -- only explain the scores given above."
    )
    try:
        result = await asyncio.to_thread(
            generate_structured,
            prompt,
            _RiskNarrative,
            system_instruction=(
                "You explain already-computed clinical risk heuristics in plain "
                "language. You never compute, alter, or invent risk scores yourself. "
                "This is a demo system, not medical advice."
            ),
        )
        return result.explanation
    except GeminiServiceError as exc:
        logger.warning("gemini_risk_narrative_failed error=%s", repr(exc))
        return None


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
        bucket = _bucket(overall)

        warnings = ["heuristic_score_not_a_validated_clinical_risk_score"]
        narrative = await _explain_risk_with_gemini(factors, bucket)
        if narrative:
            reasoning_summary = narrative
        else:
            warnings.append("gemini_narrative_unavailable_used_template_summary")
            reasoning_summary = (
                f"Computed heuristic risk scores from available signal/longitudinal/diagnostic "
                f"inputs. Overall bucket: {bucket}."
            )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=confidence,
            warnings=warnings,
            recommended_next_agents=["personalization_agent"],
            requires_human_review=overall >= 0.5,
            reasoning_summary=reasoning_summary,
            raw_payload={
                "heuristic_scores": factors,
                "overall_bucket": bucket,
                "score_type": "heuristic_score",  # explicitly not "validated_clinical_risk_score"
                "narrative_source": "gemini" if narrative else "template_fallback",
            },
        )