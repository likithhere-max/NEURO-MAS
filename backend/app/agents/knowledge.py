"""
Knowledge / Literature (RAG) Agent (spec 6.9).
Retrieves source-backed evidence from the curated neurological disease
knowledge base for the candidate conditions the diagnostic agent surfaced.
Every returned item carries its source URL and last_verified date; the agent
never invents a citation -- if nothing in the knowledge base matches, it
says so explicitly instead of fabricating a reference.

Gemini is used here the same way as in risk.py and diagnostic.py: it only
writes the plain-language reasoning_summary describing what evidence was
found. It never adds, removes, or alters a citation/source URL, and it is
never called when there is nothing to explain (no citations found).
"""
from __future__ import annotations

import asyncio

from pydantic import BaseModel

from app.agents.base import BaseNeuroAgent
from app.ai.gemini_service import GeminiServiceError, generate_structured
from app.core.messaging import EvidenceItem
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.knowledge_base import get_knowledge_base


class _EvidenceNarrative(BaseModel):
    explanation: str


async def _explain_evidence_with_gemini(evidence_summaries: list[dict]) -> str | None:
    """
    Asks Gemini to write a short plain-language summary of the evidence that
    was retrieved -- and nothing else. The prompt gives Gemini only the
    already-verified facts (condition name, source count, last verified
    date, a short definition excerpt) and explicitly forbids inventing
    anything beyond them. Returns None on any failure so the caller falls
    back to the template sentence.
    """
    if not evidence_summaries:
        return None

    lines = [
        f"- {item['condition']}: {item['source_count']} cited source(s), "
        f"last verified {item['last_verified']}. Definition: {item['definition']}"
        for item in evidence_summaries
    ]
    prompt = (
        "Here is the source-backed evidence retrieved for a set of candidate "
        "neurological conditions:\n\n" + "\n".join(lines) + "\n\n"
        "Write a short (2-4 sentence) plain-language summary of what evidence "
        "was found, for a clinician reviewing this case."
    )
    system_instruction = (
        "You summarize already-retrieved medical evidence for a clinical "
        "documentation assistant. Only describe what is explicitly listed "
        "above -- never invent a source, a statistic, a condition, or a "
        "clinical claim that isn't given to you. Do not make or imply a "
        "diagnosis. Output only the summary text."
    )

    try:
        result = await asyncio.to_thread(
            generate_structured,
            prompt,
            _EvidenceNarrative,
            system_instruction=system_instruction,
        )
        return result.explanation
    except GeminiServiceError:
        return None


class KnowledgeAgent(BaseNeuroAgent):
    name = "knowledge_agent"
    capabilities = ["evidence_retrieval", "citation"]
    reliability_weight = 0.8

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        kb = get_knowledge_base()
        payload = task.input_payload or {}
        upstream = payload.get("upstream_results", {})
        diagnostic = upstream.get("neuro_diagnostic_agent", {}).get("raw_payload", {})
        candidates = diagnostic.get("candidate_conditions", [])

        evidence: list[EvidenceItem] = []
        findings = []
        citations = []
        evidence_summaries = []

        matched_candidates = 0
        for c in candidates[:3]:
            record = kb.get(c["disease_id"])
            if not record:
                continue
            matched_candidates += 1
            # Strength = the candidate's own retrieval relevance (previously a constant 0.7 for every source).
            relevance = round(min(max(float(c.get("retrieval_score", 0.0)), 0.0), 1.0), 2)
            source_count = len(record.get("source_urls", []))
            for url in record.get("source_urls", []):
                citations.append({
                    "disease": record["preferred_name"],
                    "source_url": url,
                    "last_verified": record.get("last_verified"),
                    "review_status": "PROTOTYPE - curated summary, pending clinician verification",
                })
                evidence.append(EvidenceItem(
                    source_type="knowledge_base",
                    description=f"{record['preferred_name']}: {record.get('definition', '')[:180]}",
                    reference=url,
                    strength=relevance,
                ))
            findings.append(AgentFinding(
                label=f"evidence:{record['preferred_name']}",
                detail=f"{source_count} source(s) cited, last verified {record.get('last_verified')}",
                severity="info",
            ))
            evidence_summaries.append({
                "condition": record["preferred_name"],
                "definition": record.get("definition", ""),
                "source_count": source_count,
                "last_verified": record.get("last_verified"),
            })

        if not citations:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                confidence=0.0,
                warnings=["no_matching_knowledge_base_entry_no_citation_fabricated"],
                reasoning_summary="No candidate condition matched an entry in the curated knowledge base; "
                                   "no citation was fabricated to fill the gap.",
                raw_payload={"citations": [], "narrative_source": "not_applicable_no_citations"},
            )

        narrative = await _explain_evidence_with_gemini(evidence_summaries)
        warnings = ["knowledge_base_is_a_curated_prototype_subset_not_a_full_literature_index", "confidence_is_kb_coverage_not_probability",
                    "citations_are_url_lookups_no_passage_level_verification_of_claims"]
        if narrative:
            reasoning_summary = narrative
            narrative_source = "gemini"
        else:
            reasoning_summary = (
                f"Retrieved {len(citations)} source-backed citation(s) for "
                f"{len(candidates[:3])} candidate condition(s)."
            )
            narrative_source = "template_fallback"
            warnings.append("gemini_narrative_unavailable_used_template_summary")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            # Coverage, NOT a probability: share of the (<=3) candidates that have a curated KB entry.
            confidence=round(matched_candidates / max(len(candidates[:3]), 1), 2),
            evidence=evidence,
            warnings=warnings,
            recommended_next_agents=[],
            requires_human_review=False,
            reasoning_summary=reasoning_summary,
            raw_payload={"citations": citations, "narrative_source": narrative_source},
        )