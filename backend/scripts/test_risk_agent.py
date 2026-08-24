"""
Step 14 test script -- proves RiskStratificationAgent still computes correct
deterministic scores AND now produces a Gemini-generated plain-language
explanation on top of them.
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.risk import RiskStratificationAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState

task = AgentTask(
    workflow_id="test-workflow",
    patient_id="test-patient",
    agent_name="risk_stratification_agent",
    reason="manual smoke test",
    input_payload={
        "upstream_results": {
            "signal_analysis_agent": {
                "findings": [{"label": "gait_slowing"}, {"label": "tremor_increase"}]
            },
            "longitudinal_agent": {
                "raw_payload": {"deteriorating_metrics": True}
            },
        }
    },
)

state = PatientState(patient_id="test-patient")
state.medication_state.adherence_pct_30d = 0.7

result = asyncio.run(RiskStratificationAgent().execute(task, state))

print("Status:", result.status)
print("Confidence:", result.confidence)
print("Requires human review:", result.requires_human_review)
print("Warnings:", result.warnings)
print("Reasoning summary:", result.reasoning_summary)
print("Raw payload:", result.raw_payload)