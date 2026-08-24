"""
Step 18 verification -- proves the diagnostic agent now uses real semantic
retrieval end-to-end, by describing symptoms in a patient's own words
instead of the exact textbook terms the old lexical matcher required.
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.diagnostic import NeuroDiagnosticAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState

task = AgentTask(
    workflow_id="test-workflow",
    patient_id="test-patient",
    agent_name="neuro_diagnostic_agent",
    reason="manual smoke test -- rephrased symptoms",
    input_payload={
        "observations": [
            {"modality": "symptom", "free_text": "my hands shake a lot when I'm sitting still"},
            {"modality": "symptom", "free_text": "I've been moving really slowly lately"},
        ]
    },
)

state = PatientState(patient_id="test-patient")

result = asyncio.run(NeuroDiagnosticAgent().execute(task, state))

print("Status:", result.status)
print("Confidence:", result.confidence)
print("Warnings:", result.warnings)
print("Reasoning summary:", result.reasoning_summary)
print("Top candidate:", result.raw_payload["candidate_conditions"][0]["preferred_name"] if result.raw_payload["candidate_conditions"] else "none")
print("Retrieval method:", result.raw_payload["retrieval_method"])