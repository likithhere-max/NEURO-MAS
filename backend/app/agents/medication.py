"""
Medication / Drug Safety Agent (spec 6.7).

Runs deterministic interaction, contraindication and allergy checks against
the patient's active medication list. CRITICAL constraint enforced here in
code, not just docs: this agent NEVER proposes a dose change -- its only
output type is a "medication safety finding" that always carries
requires_human_review=True.
"""
from __future__ import annotations

from app.agents.base import BaseNeuroAgent
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.clinical_rules import check_allergy_conflicts, check_interactions


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

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=confidence,
            warnings=["curated_interaction_table_is_illustrative_not_exhaustive"],
            recommended_next_agents=["risk_stratification_agent"],
            requires_human_review=True,  # ALL medication safety findings require clinician review, always
            reasoning_summary=(
                f"Checked {len(active_meds)} active medication(s) against a curated interaction/"
                f"allergy table. Found {len(interactions)} interaction(s), {len(allergy_hits)} "
                f"allergy conflict(s). No dose changes are proposed by this agent under any circumstance."
            ),
            raw_payload={
                "interactions": interactions,
                "allergy_conflicts": allergy_hits,
                "adherence_pct_30d": adherence,
                "action_restriction": "MEDICATION_CHANGES_REQUIRE_CLINICIAN_AUTHORIZATION",
            },
        )
