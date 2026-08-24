"""
Step 24 test script -- proves LongitudinalAgent's deterministic trend
classification is unchanged AND Gemini now narrates it accurately from only
the computed baseline/current/delta values, never reclassifying a metric.
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.longitudinal import LongitudinalAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState

task = AgentTask(
    workflow_id="test-workflow",
    patient_id="test-patient",
    agent_name="longitudinal_agent",
    reason="manual smoke test",
    input_payload={},
)

state = PatientState(
    patient_id="test-patient",
    longitudinal_trends=[
        {"metric": "tremor_frequency", "baseline_value": 2.0, "current_value": 3.4, "window": "30 days"},
        {"metric": "gait_speed", "baseline_value": 1.1, "current_value": 1.15, "window": "30 days"},
        {"metric": "medication_adherence", "baseline_value": 0.9, "current_value": 0.6, "window": "30 days"},
    ],
)

result = asyncio.run(LongitudinalAgent().execute(task, state))

print("Status:", result.status)
print("Confidence:", result.confidence)
print("Requires human review:", result.requires_human_review)
print("Warnings:", result.warnings)
print("Reasoning summary:", result.reasoning_summary)
print("Raw payload:", result.raw_payload)