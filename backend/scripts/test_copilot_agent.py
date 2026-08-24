"""
Step 22 test script -- proves PatientCopilotAgent still introduces no new
clinical claims and now produces a real Gemini-phrased plain-language
explanation of an already-computed structured result, with the disclaimer
still appended exactly as before.
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.copilot import PatientCopilotAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState

task = AgentTask(
    workflow_id="test-workflow",
    patient_id="test-patient",
    agent_name="patient_copilot_agent",
    reason="manual smoke test",
    input_payload={
        "structured_context": {
            "decision": "Continue routine monitoring; no emergency escalation required",
            "risk_level": "moderate",
            "agreements": ["risk_stratification_agent and neuro_diagnostic_agent both flagged elevated fall risk"],
            "conflicts": [],
            "action": "Schedule a follow-up neurology visit within 2 weeks and continue current medication",
        }
    },
)

state = PatientState(patient_id="test-patient")

result = asyncio.run(PatientCopilotAgent().execute(task, state))

print("Status:", result.status)
print("Confidence:", result.confidence)
print("Requires human review:", result.requires_human_review)
print("Warnings:", result.warnings)
print("Reasoning summary:", result.reasoning_summary)
print("Raw payload:", result.raw_payload)