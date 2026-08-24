import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.personalization import PersonalizationAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState

task = AgentTask(
    workflow_id="test-workflow",
    patient_id="test-patient",
    agent_name="personalization_agent",
    reason="manual smoke test",
    input_payload={
        "upstream_results": {
            "risk_stratification_agent": {
                "raw_payload": {
                    "heuristic_scores": {"fall_risk": 0.65, "seizure_risk": 0.2}
                }
            },
            "medication_safety_agent": {
                "raw_payload": {"adherence_pct_30d": 0.6}
            },
            "longitudinal_agent": {
                "raw_payload": {
                    "deteriorating_metrics": ["tremor_frequency"],
                    "improving_metrics": ["gait_speed"],
                }
            },
        }
    },
)

state = PatientState(patient_id="test-patient")
state.disease_state.conditions = ["Parkinson's disease"]
state.patient_preferences.preferred_language = "Hindi"

result = asyncio.run(PersonalizationAgent().execute(task, state))

print("status:", result.status)
print("requires_human_review:", result.requires_human_review)
print("warnings:", result.warnings)
print("narrative_source:", result.raw_payload["narrative_source"])
print()
print("reasoning_summary:")
print(result.reasoning_summary)
print()
print("monitoring_plan:", result.raw_payload["monitoring_plan"])
print("educational_content:", result.raw_payload["educational_content"])