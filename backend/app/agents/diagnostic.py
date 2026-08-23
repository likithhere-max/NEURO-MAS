"""
Neuro-Diagnostic Agent (spec 6.3).

Performs symptom-pattern matching against the curated neurological disease
knowledge base. This is explicitly a lexical-overlap heuristic, NOT a
trained diagnostic classifier -- confidence is derived from term overlap and
capped conservatively so the system never "claims certainty from
insufficient information," per the spec's own constraint on this agent.
"""
from __future__ import annotations

from app.agents.base import BaseNeuroAgent
from app.core.messaging import EvidenceItem
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.knowledge_base import get_knowledge_base

MAX_HEURISTIC_CONFIDENCE = 0.75  # never let a keyword-overlap score claim near-certainty


class NeuroDiagnosticAgent(BaseNeuroAgent):
    name = "neuro_diagnostic_agent"
    capabilities = ["differential_diagnosis", "syndrome_matching"]
    reliability_weight = 0.65  # heuristic lexical matcher -> moderate trust

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        kb = get_knowledge_base()
        observations = (task.input_payload or {}).get("observations", [])
        symptom_terms = [
            o.get("name") or o.get("free_text", "")
            for o in observations
            if o.get("modality") == "symptom"
        ]
        symptom_terms = [t for t in symptom_terms if t]

        known_conditions = state.disease_state.conditions or []

        if not symptom_terms:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                confidence=0.0,
                warnings=["no_symptom_data"],
                requires_human_review=False,
                reasoning_summary="No symptom observations were provided; diagnostic matching skipped.",
                raw_payload={"candidate_conditions": [], "missing_information": ["symptoms"]},
            )

        matches = kb.match_symptoms(symptom_terms, top_k=5)

        candidates = []
        evidence: list[EvidenceItem] = []
        contradicting: list[str] = []
        red_flags: list[str] = []

        for m in matches:
            confidence = min(m["_match_score"], 1.0) * MAX_HEURISTIC_CONFIDENCE
            # Boost slightly if this condition is already part of the patient's
            # known history -- prior diagnosis is real evidence, not noise.
            if m["preferred_name"] in known_conditions or m["disease_id"] in known_conditions:
                confidence = min(confidence + 0.15, MAX_HEURISTIC_CONFIDENCE)

            candidates.append({
                "disease_id": m["disease_id"],
                "preferred_name": m["preferred_name"],
                "confidence": round(confidence, 2),
                "matched_terms": m["_matched_terms"],
                "icd10_codes": m.get("icd10_codes", []),
            })
            evidence.append(EvidenceItem(
                source_type="knowledge_base",
                description=f"Symptom overlap with {m['preferred_name']}: {', '.join(m['_matched_terms'])}",
                reference=(m.get("source_urls") or [None])[0],
                strength=round(m["_match_score"], 2),
            ))
            red_flags.extend(m.get("red_flags", []) or [])

        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        top_confidence = candidates[0]["confidence"] if candidates else 0.0

        missing_info = []
        if not observations or len(observations) < 2:
            missing_info.append("Limited observation set; consider requesting EEG/EMG, imaging, or lab workup per differential.")

        findings = [
            AgentFinding(
                label=f"candidate:{c['preferred_name']}",
                detail=f"confidence={c['confidence']}, matched={c['matched_terms']}",
                severity="info",
                value=c["confidence"],
            )
            for c in candidates
        ]

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=round(top_confidence, 2),
            evidence=evidence,
            warnings=["heuristic_lexical_matching_only_not_a_clinical_diagnosis"],
            recommended_next_agents=["risk_stratification_agent", "longitudinal_agent"],
            requires_human_review=True,  # diagnostic output ALWAYS requires clinician review
            reasoning_summary=(
                f"Matched {len(candidates)} candidate condition(s) via symptom-term overlap "
                f"against the curated knowledge base. Top candidate confidence capped at "
                f"{MAX_HEURISTIC_CONFIDENCE} to reflect that this is a lexical heuristic, not a "
                f"validated diagnostic model."
            ),
            raw_payload={
                "candidate_conditions": candidates,
                "contradicting_evidence": contradicting,
                "missing_information": missing_info,
                "red_flags": sorted(set(red_flags)),
            },
        )
