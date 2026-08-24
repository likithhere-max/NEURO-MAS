"""
Medication / Drug Safety Agent (spec 6.7).
Runs deterministic interaction, contraindication and allergy checks against
the patient's active medication list. CRITICAL constraint enforced here in
code, not just docs: this agent NEVER proposes a dose change -- its only
output type is a "medication safety finding" that always carries
requires_human_review=True.

Gemini narrates ONLY the already-computed findings (interaction pairs,
allergy conflicts, adherence percentage) -- it never adds a finding, never
changes a severity, and is explicitly forbidden from suggesting or implying
any medication action. requires_human_review stays hardcoded True regardless
of narrative success/failure, unconditionally, exactly as before.
"""
from __future__ import annotations

import asyncio

from pydantic import BaseModel

from app.agents.base import BaseNeuroAgent
from app.ai.gemini_service import GeminiServiceError, generate_structured
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.clinical_rules import check_allergy_conflicts, check_interactions


class _MedicationSafetyNarrative(BaseModel):
    explanation: str


async def _explain_medication_safety_with_gemini(
    active_med_count: int,
    interactions: list[dict],
    allergy_hits: list[dict],
    adherence: float | None,
) -> str | None:
    """
    Gives Gemini ONLY the already-computed findings -- nothing else -- so it
    has nothing to fabricate. Returns None on any failure so the caller
    falls back to the template.
    """
    lines = [f"- {active_med_count} active medication(s) checked."]
    if interactions:
        for i in interactions:
            lines.append(f"- Interaction found: {i['drug_a']} + {i['drug_b']} ({i['severity']} severity): {i['description']}")
    else:
        lines.append("- No known drug-drug interactions found in the curated table.")
    if allergy_hits:
        for a in allergy_hits:
            lines.append(f"- Allergy conflict found: patient is on {a['drug']}, which is a documented allergy ({a['severity']} severity).")
    else:
        lines.append("- No allergy conflicts found.")
    if adherence is not None:
        lines.append(f"- 30-day medication adherence estimated at {round(adherence * 100, 1)}%.")

    prompt = (
        "Here is an already-computed, deterministic medication safety check result:\n\n"
        + "\n".join(lines) + "\n\n"
        "Write a short (2-4 sentence) plain-language summary of these findings "
        "for a clinician reviewing this patient's medications."
    )
    system_instruction = (
        "You summarize already-computed medication safety check results for a "
        "clinical documentation assistant. Only describe what is explicitly "
        "listed above -- never invent an interaction, an allergy, a drug, or "
        "a value not given to you. You must NEVER propose, suggest, or imply "
        "any medication dose change, addition, discontinuation, or other "
        "action, under any circumstance -- that decision belongs to a "
        "clinician, never to you. Output only the summary text."
    )
    try:
        result = await asyncio.to_thread(
            generate_structured,
            prompt,
            _MedicationSafetyNarrative,
            system_instruction=system_instruction,
        )
        return result.explanation
    except GeminiServiceError:
        return None


class MedicationSafetyAgent(BaseNeuroAgent):
    name = "medication_safety_agent"
    capabilities = ["interaction_check", "allergy_check", "adherence_check"]
    reliability_weight = 0.85  # deterministic rule-table lookups -> high trust for what it covers

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        active_meds = [m.get("name") for m in state.medication_state.active_medications if m.get("name")]
        allergies = state.medication_state.allergies or []

        findings = []
        interactions = check_interactions(active_meds)
        for i in interactions:
            findings.append(AgentFinding(
                label=f"interaction:{i['drug_a']}+{i['drug_b']}",
                detail=i["description"],
                severity=i["severity"],
            ))

        allergy_hits = check_allergy_conflicts(active_meds, allergies)
        for a in allergy_hits:
            findings.append(AgentFinding(
                label=f"allergy_conflict:{a['drug']}",
                detail=a["description"],
                severity=a["severity"],
            ))

        adherence = state.medication_state.adherence_pct_30d
        if adherence is not None and adherence < 0.8:
            findings.append(AgentFinding(
                label="adherence_below_threshold",
                detail=f"30-day adherence estimated at {round(adherence*100,1)}%, below the 80% monitoring threshold.",
                severity="medium",
                value=adherence,
            ))

        has_findings = bool(findings)
        confidence = 0.8 if has_findings else 0.9  # confident either way; this is a table lookup

        template_summary = (
            f"Checked {len(active_meds)} active medication(s) against a curated interaction/"
            f"allergy table. Found {len(interactions)} interaction(s), {len(allergy_hits)} "
            f"allergy conflict(s). No dose changes are proposed by this agent under any circumstance."
        )

        warnings = ["curated_interaction_table_is_illustrative_not_exhaustive"]
        narrative = await _explain_medication_safety_with_gemini(
            len(active_meds), interactions, allergy_hits, adherence,
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
            confidence=confidence,
            warnings=warnings,
            recommended_next_agents=["risk_stratification_agent"],
            requires_human_review=True,  # ALL medication safety findings require clinician review, always
            reasoning_summary=reasoning_summary,
            raw_payload={
                "interactions": interactions,
                "allergy_conflicts": allergy_hits,
                "adherence_pct_30d": adherence,
                "action_restriction": "MEDICATION_CHANGES_REQUIRE_CLINICIAN_AUTHORIZATION",
                "narrative_source": narrative_source,
            },
        )