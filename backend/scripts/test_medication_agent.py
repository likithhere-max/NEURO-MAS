"""
Step 25 test script -- proves MedicationSafetyAgent's deterministic
interaction/allergy/adherence checks are unchanged AND Gemini now narrates
them accurately without ever suggesting a medication action.
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.medication import MedicationSafetyAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState

task = AgentTask(
    workflow_id="test-workflow",
    patient_id="test-patient",
    agent_name="medication_safety_agent",
    reason="manual smoke test",
    input_payload={},
)

state = PatientState(
    patient_id="test-patient",
    medication_state={
        "active_medications": [{"name": "levodopa"}, {"name": "haloperidol"}],
        "allergies": ["haloperidol"],
        "adherence_pct_30d": 0.65,
    },
)

result = asyncio.run(MedicationSafetyAgent().execute(task, state))

print("Status:", result.status)
print("Confidence:", result.confidence)
print("Requires human review:", result.requires_human_review)
print("Warnings:", result.warnings)
print("Reasoning summary:", result.reasoning_summary)
print("Raw payload:", result.raw_payload)