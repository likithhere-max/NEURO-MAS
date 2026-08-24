"""
Neuro-Diagnostic Agent (spec 6.3).

Performs symptom-pattern matching against the curated neurological disease
knowledge base. This is explicitly a lexical-overlap heuristic, NOT a
trained diagnostic classifier -- confidence is derived from term overlap and
capped conservatively so the system never "claims certainty from
insufficient information," per the spec's own constraint on this agent.

Gemini's role here is deliberately narrow, same principle as the risk agent:
it only turns the already-computed candidate list into a plain-language
explanation. It never adds a candidate, never changes a confidence score,
and never states a definitive diagnosis -- `requires_human_review` stays
hardcoded True regardless of what Gemini says.
"""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from app.agents.base import BaseNeuroAgent
from app.ai.gemini_service import GeminiServiceError, generate_structured
from app.core.messaging import EvidenceItem
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.knowledge_base import get_knowledge_base

logger = logging.getLogger("neuromas.agents.diagnostic")

MAX_HEURISTIC_CONFIDENCE = 0.75  # never let a keyword-overlap score claim near-certainty


class _DiagnosisNarrative(BaseModel):
    """Gemini's structured output -- ONLY a narrative explanation of the
    candidate list we already computed. No numeric or candidate-list field
    exists on this schema, so Gemini has nothing to fabricate a diagnosis
    or confidence score into."""
    explanation: str


async def _explain_diagnosis_with_gemini(candidates: list[dict], red_flags: list[str]) -> str | None:
    """
    Best-effort: turns the already-computed, deterministic candidate list
    into a clinician-readable explanation. Returns None on any failure so
    the caller falls back to the plain template sentence -- this must never
    crash the agent or cause it to invent a candidate or confidence number.
    """
    candidate_lines = "\n".join(
        f"- {c['preferred_name']} (confidence={c['confidence']}, "
        f"matched symptoms: {', '.join(c['matched_terms']) or 'none'})"
        for c in candidates
    )
    red_flag_line = f"Red flags noted: {', '.join(red_flags)}." if red_flags else "No red flags noted."

    prompt = (
        f"A lexical symptom-matching heuristic (not you) produced these candidate neurological "
        f"conditions, ranked by confidence (capped at {MAX_HEURISTIC_CONFIDENCE} since this is "
        f"keyword overlap, not a validated diagnostic model):\n{candidate_lines}\n\n{red_flag_line}\n\n"
        "Write a 2-4 sentence plain-language explanation for a clinician of what this candidate "
        "list suggests and why, based only on the information above. Do not add new candidate "
        "conditions, do not change or invent confidence numbers, and do not state a definitive "
        "diagnosis -- always make clear this requires clinician review."
    )
    try:
        result = await asyncio.to_thread(
            generate_structured,
            prompt,
            _DiagnosisNarrative,
            system_instruction=(
                "You explain already-computed differential-diagnosis candidates from a lexical "
                "matching heuristic in plain language for a clinician. You never add new "
                "candidates, never change confidence scores, and never assert a definitive "
                "diagnosis. This is a demo system, not medical advice, and all output requires "
                "clinician review."
            ),
        )
        return result.explanation
    except GeminiServiceError as exc:
        logger.warning("gemini_diagnosis_narrative_failed error=%s", repr(exc))
        return None


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

        template_summary = (
            f"Matched {len(candidates)} candidate condition(s) via symptom-term overlap "
            f"against the curated knowledge base. Top candidate confidence capped at "
            f"{MAX_HEURISTIC_CONFIDENCE} to reflect that this is a lexical heuristic, not a "
            f"validated diagnostic model."
        )
        warnings = ["heuristic_lexical_matching_only_not_a_clinical_diagnosis"]

        reasoning_summary = template_summary
        narrative_source = "template_fallback"
        if candidates:
            narrative = await _explain_diagnosis_with_gemini(candidates, sorted(set(red_flags)))
            if narrative:
                reasoning_summary = narrative
                narrative_source = "gemini"
            else:
                warnings.append("gemini_narrative_unavailable_used_template_summary")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=round(top_confidence, 2),
            evidence=evidence,
            warnings=warnings,
            recommended_next_agents=["risk_stratification_agent", "longitudinal_agent"],
            requires_human_review=True,  # diagnostic output ALWAYS requires clinician review
            reasoning_summary=reasoning_summary,
            raw_payload={
                "candidate_conditions": candidates,
                "contradicting_evidence": contradicting,
                "missing_information": missing_info,
                "red_flags": sorted(set(red_flags)),
                "narrative_source": narrative_source,
            },
        )