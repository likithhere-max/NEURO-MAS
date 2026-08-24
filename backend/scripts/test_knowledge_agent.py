"""
Step 20 test script -- proves KnowledgeAgent still retrieves the correct
source-backed citations for candidate conditions (unchanged, deterministic)
AND now produces a Gemini-generated plain-language explanation summarizing
that evidence, never inventing a citation.
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.knowledge import KnowledgeAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState

task = AgentTask(
    workflow_id="test-workflow",
    patient_id="test-patient",
    agent_name="knowledge_agent",
    reason="manual smoke test",
    input_payload={
        "upstream_results": {
            "neuro_diagnostic_agent": {
                "raw_payload": {
                    "candidate_conditions": [
                        {"disease_id": "NEURO-PD-002", "preferred_name": "Parkinson's disease", "confidence": 0.75},
                    ]
                }
            }
        }
    },
)

state = PatientState(patient_id="test-patient")

result = asyncio.run(KnowledgeAgent().execute(task, state))

print("Status:", result.status)
print("Confidence:", result.confidence)
print("Requires human review:", result.requires_human_review)
print("Warnings:", result.warnings)
print("Reasoning summary:", result.reasoning_summary)
print("Evidence:", result.evidence)
print("Raw payload:", result.raw_payload)