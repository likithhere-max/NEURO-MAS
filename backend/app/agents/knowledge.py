"""
Knowledge / Literature (RAG) Agent (spec 6.9).

Retrieves source-backed evidence from the curated neurological disease
knowledge base for the candidate conditions the diagnostic agent surfaced.
Every returned item carries its source URL and last_verified date; the agent
never invents a citation -- if nothing in the knowledge base matches, it
says so explicitly instead of fabricating a reference.
"""
from __future__ import annotations

from app.agents.base import BaseNeuroAgent
from app.core.messaging import EvidenceItem
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.knowledge_base import get_knowledge_base


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

        for c in candidates[:3]:
            record = kb.get(c["disease_id"])
            if not record:
                continue
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
                    strength=0.7,
                ))
            findings.append(AgentFinding(
                label=f"evidence:{record['preferred_name']}",
                detail=f"{len(record.get('source_urls', []))} source(s) cited, last verified {record.get('last_verified')}",
                severity="info",
            ))

        if not citations:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                confidence=0.0,
                warnings=["no_matching_knowledge_base_entry_no_citation_fabricated"],
                reasoning_summary="No candidate condition matched an entry in the curated knowledge base; "
                                   "no citation was fabricated to fill the gap.",
                raw_payload={"citations": []},
            )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=0.75,
            evidence=evidence,
            warnings=["knowledge_base_is_a_curated_prototype_subset_not_a_full_literature_index"],
            recommended_next_agents=[],
            requires_human_review=False,
            reasoning_summary=f"Retrieved {len(citations)} source-backed citation(s) for {len(candidates[:3])} candidate condition(s).",
            raw_payload={"citations": citations},
        )
