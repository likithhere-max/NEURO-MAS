import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.signal_analysis import SignalAnalysisAgent
from app.schemas.agent import AgentTask
from app.schemas.patient_state import PatientState


def run(label, wearable):
    task = AgentTask(
        workflow_id="test-workflow",
        patient_id="test-patient",
        agent_name="signal_analysis_agent",
        reason="manual smoke test",
        input_payload={"wearable": wearable},
    )
    state = PatientState(patient_id="test-patient")
    state.wearable_metrics = {
        "tremor_amplitude_baseline": 2.0,
        "gait_speed_baseline": 1.2,
    }
    result = asyncio.run(SignalAnalysisAgent().execute(task, state))
    print(f"--- {label} ---")
    print("status:", result.status)
    print("requires_human_review:", result.requires_human_review)
    print("warnings:", result.warnings)
    print("narrative_source:", result.raw_payload.get("narrative_source"))
    print("reasoning_summary:", result.reasoning_summary)
    print()


run("findings + unsupported modality", {
    "tremor_amplitude": 3.5,
    "gait_speed_m_s": 0.9,
    "eeg_raw": {"note": "raw eeg blob placeholder"},
})

run("no wearable data at all", {})