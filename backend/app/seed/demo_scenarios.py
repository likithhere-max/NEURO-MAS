"""
Runnable end-to-end demo scenarios (spec section 38).

Usage:
    python -m app.seed.demo_scenarios

Creates demo patients (idempotent-ish -- reruns add new demo patients each
time; fine for a prototype) and runs full multi-agent workflows against
them, printing the resulting consensus, conflicts, and proposed actions so
you can see the whole pipeline fire without needing the frontend up.
"""
from __future__ import annotations

import asyncio
import json

from app.db.models import Allergy, Condition, LongitudinalMetric, Medication, Patient
from app.db.session import Base, SessionLocal, engine
from app.orchestrator.orchestrator import run_workflow
from app.services.repository import build_patient_state, persist_workflow


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def scenario_epilepsy_emergency(db):
    """Scenario 2 from the spec: unusual event report -> possible seizure,
    with a deterministic emergency trigger to demonstrate Level 4 escalation.
    """
    patient = Patient(full_name="Demo Patient - Epilepsy (Asha)", age=29, sex="female", preferred_language="en")
    db.add(patient); db.flush()
    db.add(Condition(patient_id=patient.id, disease_id="NEURO-EPI-001", preferred_name="Epilepsy (focal and generalized)"))
    db.add(Medication(patient_id=patient.id, name="levetiracetam", active=True, adherence_pct_30d=0.65))
    db.add(LongitudinalMetric(patient_id=patient.id, metric="seizure_free_days", baseline_value=45, current_value=2, window="rolling"))
    db.commit()

    state = build_patient_state(db, patient)
    result = await run_workflow(
        patient_id=patient.id,
        trigger_reason="patient_reported_unusual_event",
        input_payload={
            "symptoms": [
                {"name": "seizure", "severity": "severe", "free_text": "seizure not stopping, status epilepticus, over 5 minutes and not regaining consciousness"},
            ],
            "medication_logs": [{"name": "levetiracetam", "taken": False, "scheduled_time": "08:00"}],
            "wearable": {},
            "vitals": {},
            "caregiver_notes": "Caregiver reports seizure has continued for several minutes without stopping.",
        },
        patient_state=state,
    )
    persist_workflow(db, result)
    _print_header(f"SCENARIO: Epilepsy emergency -- patient {patient.full_name}")
    _print_summary(result)
    return patient, result


async def scenario_parkinsons_deterioration(db):
    """Scenario 1 from the spec: tremor increase + gait slowing + adherence
    issue -> multi-agent agreement -> personalized action, non-emergency.
    """
    patient = Patient(full_name="Demo Patient - Parkinson's (Rahul)", age=67, sex="male", preferred_language="en")
    db.add(patient); db.flush()
    db.add(Condition(patient_id=patient.id, disease_id="NEURO-PD-002", preferred_name="Parkinson's disease"))
    db.add(Medication(patient_id=patient.id, name="levodopa", active=True, adherence_pct_30d=0.72))
    db.add(LongitudinalMetric(patient_id=patient.id, metric="tremor_amplitude", baseline_value=1.0, current_value=1.0, window="rolling_7d"))
    db.add(LongitudinalMetric(patient_id=patient.id, metric="gait_speed_m_s", baseline_value=1.1, current_value=1.1, window="rolling_7d"))
    db.commit()

    state = build_patient_state(db, patient)
    result = await run_workflow(
        patient_id=patient.id,
        trigger_reason="wearable_trend_deviation",
        input_payload={
            "symptoms": [{"name": "tremor", "severity": "moderate", "free_text": "increased hand tremor and slower walking today"}],
            "medication_logs": [{"name": "levodopa", "taken": False, "scheduled_time": "07:00"}],
            "wearable": {"tremor_amplitude": 1.7, "gait_speed_m_s": 0.85},
            "vitals": {},
            "caregiver_notes": None,
        },
        patient_state=state,
    )
    persist_workflow(db, result)
    _print_header(f"SCENARIO: Parkinson's deterioration -- patient {patient.full_name}")
    _print_summary(result)
    return patient, result


def _print_summary(result) -> None:
    print(f"workflow_id: {result.workflow_id}")
    print(f"is_emergency: {result.is_emergency}")
    print("\nAgent results:")
    for name, r in result.agent_results.items():
        print(f"  - {name}: status={r.status} confidence={r.confidence} :: {r.reasoning_summary}")
    if result.consensus:
        print(f"\nConsensus: top={result.consensus.top_hypothesis} disagreement={result.consensus.disagreement}")
        print(f"  {result.consensus.reason}")
    print("\nProposed actions:")
    for a in result.proposed_actions:
        print(f"  - [{a.status}] {a.action_type} (autonomy_level={a.autonomy_level}, requires_approval={a.requires_human_approval})")
        print(f"      reason: {a.reason}")
        print(f"      safety: {a.safety_reasons}")


async def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        await scenario_epilepsy_emergency(db)
        await scenario_parkinsons_deterioration(db)
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
