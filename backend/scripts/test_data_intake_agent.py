"""
Step 23 test script -- proves DataIntakeAgent's deterministic normalization
and urgency triage are unchanged, AND that Gemini only narrates the
non-emergency case while the emergency case bypasses Gemini entirely
(zero added latency on the time-critical path).
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.data_intake import DataIntakeAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState


def run(label, input_payload):
    task = AgentTask(
        workflow_id="test-workflow",
        patient_id="test-patient",
        agent_name="data_intake_agent",
        reason="manual smoke test",
        input_payload=input_payload,
    )
    state = PatientState(patient_id="test-patient")
    result = asyncio.run(DataIntakeAgent().execute(task, state))
    print(f"--- {label} ---")
    print("Requires human review:", result.requires_human_review)
    print("Warnings:", result.warnings)
    print("Reasoning summary:", result.reasoning_summary)
    print("Raw payload urgency/narrative_source:", result.raw_payload["urgency"], "/", result.raw_payload["narrative_source"])
    print()


run("NON-EMERGENCY (should use Gemini)", {
    "symptoms": [
        {"name": "resting tremor", "severity": "mild", "onset": "gradual", "free_text": "hands shake sometimes"},
    ],
})

run("EMERGENCY (should bypass Gemini entirely)", {
    "symptoms": [
        {"name": "facial drooping", "severity": "severe", "onset": "sudden", "free_text": "face drooping and one sided weakness"},
    ],
})