"""
Patient Data / Intake Agent (spec 6.2).
Normalizes raw, heterogeneous patient inputs (symptom reports, questionnaires,
medication logs, wearable readings, caregiver notes) into structured
observations, flags missing data, and produces a first-pass urgency
classification using the deterministic emergency keyword scan -- NOT an LLM
judgment call.

Gemini is used ONLY to narrate the reasoning_summary, and ONLY when the
deterministic scan found no emergency. When is_emergency is True, Gemini is
never called -- the original deterministic template sentence is used
directly, so the one path where speed matters most never waits on an
external API call, matching the same reasoning applied to
emergency_escalation_agent.
"""
from __future__ import annotations

import asyncio

from pydantic import BaseModel

from app.agents.base import BaseNeuroAgent
from app.ai.gemini_service import GeminiServiceError, generate_structured
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.clinical_rules import detect_emergency

EXPECTED_FIELDS = ["symptoms", "medication_logs", "wearable", "vitals", "caregiver_notes"]


class _IntakeNarrative(BaseModel):
    explanation: str


async def _explain_intake_with_gemini(
    observation_count: int,
    present_count: int,
    expected_total: int,
    missing: list[str],
    urgency: str,
    data_quality: float,
) -> str | None:
    """
    Only called when is_emergency is False. Gives Gemini exactly the
    already-computed facts and nothing else, so it has nothing to fabricate.
    Returns None on any failure so the caller falls back to the template.
    """
    prompt = (
        f"A patient data intake was just normalized:\n"
        f"- {observation_count} structured observation(s) captured\n"
        f"- {present_count}/{expected_total} expected data modalities present "
        f"(missing: {', '.join(missing) if missing else 'none'})\n"
        f"- Data quality score: {data_quality}\n"
        f"- Urgency classified as: {urgency}\n\n"
        "Write a short (1-3 sentence) plain-language summary of this intake "
        "for a clinician reviewing it."
    )
    system_instruction = (
        "You summarize an already-computed, deterministic data-intake result "
        "for a clinician. Only restate the facts given above -- never invent "
        "additional missing fields, observations, or a clinical "
        "interpretation not present above. Never suggest a diagnosis or "
        "treatment. Output only the summary text."
    )
    try:
        result = await asyncio.to_thread(
            generate_structured,
            prompt,
            _IntakeNarrative,
            system_instruction=system_instruction,
        )
        return result.explanation
    except GeminiServiceError:
        return None


class DataIntakeAgent(BaseNeuroAgent):
    name = "data_intake_agent"
    capabilities = ["normalization", "validation", "urgency_triage"]
    reliability_weight = 0.9  # deterministic, not model-based -> high trust

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        raw = task.input_payload or {}
        observations: list[dict] = []
        missing: list[str] = []

        symptoms = raw.get("symptoms", [])
        for s in symptoms:
            observations.append({
                "modality": "symptom",
                "name": s.get("name"),
                "severity": s.get("severity"),
                "onset": s.get("onset"),
                "free_text": s.get("free_text", ""),
            })
        if not symptoms:
            missing.append("symptoms")

        med_logs = raw.get("medication_logs", [])
        for m in med_logs:
            observations.append({
                "modality": "medication_log",
                "name": m.get("name"),
                "taken": m.get("taken"),
                "scheduled_time": m.get("scheduled_time"),
            })
        if not med_logs:
            missing.append("medication_logs")

        wearable = raw.get("wearable", {})
        if wearable:
            observations.append({"modality": "wearable", **wearable})
        else:
            missing.append("wearable")

        vitals = raw.get("vitals", {})
        if vitals:
            observations.append({"modality": "vital", **vitals})
        else:
            missing.append("vitals")

        caregiver_notes = raw.get("caregiver_notes")
        if caregiver_notes:
            observations.append({"modality": "caregiver_report", "note": caregiver_notes})
        else:
            missing.append("caregiver_notes")

        text_blob = [s.get("free_text", "") or s.get("name", "") for s in symptoms]
        if caregiver_notes:
            text_blob.append(caregiver_notes)
        is_emergency, matched = detect_emergency(text_blob)

        data_quality = round(1 - (len(missing) / len(EXPECTED_FIELDS)), 2)

        if is_emergency:
            urgency = "critical"
        elif any((s.get("severity") or "").lower() in {"severe", "high"} for s in symptoms):
            urgency = "high"
        elif symptoms:
            urgency = "medium"
        else:
            urgency = "low"

        findings = [
            AgentFinding(label="observations_captured", detail=f"{len(observations)} structured observations", value=len(observations)),
            AgentFinding(label="urgency", detail=urgency, severity=urgency if urgency != "low" else "info"),
        ]
        if matched:
            findings.append(AgentFinding(
                label="emergency_keyword_match",
                detail=f"Deterministic keyword scan matched: {', '.join(matched)}",
                severity="critical",
            ))

        present_count = len(EXPECTED_FIELDS) - len(missing)
        template_summary = (
            f"Normalized {len(observations)} observations across "
            f"{present_count}/{len(EXPECTED_FIELDS)} expected modalities. "
            f"Urgency classified as '{urgency}' via deterministic rules."
        )

        warnings = [f"missing_data:{m}" for m in missing]

        if is_emergency:
            # Zero API dependency on the time-critical path -- same reasoning
            # as emergency_escalation_agent.
            reasoning_summary = template_summary
            narrative_source = "not_applicable_emergency_bypass"
        else:
            narrative = await _explain_intake_with_gemini(
                observation_count=len(observations),
                present_count=present_count,
                expected_total=len(EXPECTED_FIELDS),
                missing=missing,
                urgency=urgency,
                data_quality=data_quality,
            )
            if narrative:
                reasoning_summary = narrative
                narrative_source = "gemini"
            else:
                reasoning_summary = template_summary
                narrative_source = "template_fallback"
                warnings.append("gemini_narrative_unavailable_used_template_summary")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=0.95,  # deterministic normalization, not a probabilistic judgment
            warnings=warnings,
            recommended_next_agents=["neuro_diagnostic_agent", "risk_stratification_agent"],
            requires_human_review=is_emergency,
            reasoning_summary=reasoning_summary,
            raw_payload={
                "observations": observations,
                "data_quality": data_quality,
                "missing_data": missing,
                "urgency": urgency,
                "narrative_source": narrative_source,
            },
        )