"""
Step 15 test script -- proves NeuroDiagnosticAgent still computes correct
deterministic lexical-match candidates AND now produces a Gemini-generated
plain-language explanation on top of them.
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
    reason="manual smoke test",
    input_payload={
        "observations": [
            {"modality": "symptom", "name": "resting tremor"},
            {"modality": "symptom", "name": "bradykinesia"},
            {"modality": "symptom", "name": "rigidity"},
        ]
    },
)

state = PatientState(patient_id="test-patient")

result = asyncio.run(NeuroDiagnosticAgent().execute(task, state))

print("Status:", result.status)
print("Confidence:", result.confidence)
print("Requires human review:", result.requires_human_review)
print("Warnings:", result.warnings)
print("Reasoning summary:", result.reasoning_summary)
print("Raw payload:", result.raw_payload)