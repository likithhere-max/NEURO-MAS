"""
Repository layer: builds the in-memory PatientState (the agents' shared
blackboard) from persisted DB rows, and persists workflow outcomes back to
the DB (observations, metrics, alerts, actions, audit events).

Keeping this translation in one place means the agents/orchestrator never
touch SQLAlchemy directly -- they only ever see the Pydantic PatientState /
AgentResult contracts, which keeps the "core" package DB-agnostic.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db import models as m
from app.orchestrator.orchestrator import WorkflowState
from app.schemas.patient_state import (
    AdherenceState,
    BehavioralState,
    ClinicianConstraints,
    CognitiveState,
    Demographics,
    DiseaseState,
    EnvironmentalState,
    LongitudinalTrend,
    MedicationState,
    PatientPreferences,
    PatientState,
    RiskState,
    SymptomState,
)


def build_patient_state(db: Session, patient: m.Patient) -> PatientState:
    conditions = [c.preferred_name for c in patient.conditions]
    active_meds = [
        {"name": med.name, "active": med.active, "schedule": med.schedule}
        for med in patient.medications if med.active
    ]
    adherence_values = [med.adherence_pct_30d for med in patient.medications if med.adherence_pct_30d is not None]
    adherence_avg = round(sum(adherence_values) / len(adherence_values), 3) if adherence_values else None

    trends = [
        LongitudinalTrend(
            metric=met.metric,
            baseline_value=met.baseline_value,
            current_value=met.current_value,
            window=met.window,
            trend=(
                "unknown" if met.baseline_value is None or met.current_value is None
                else "deteriorating" if met.current_value > met.baseline_value * 1.05
                else "improving" if met.current_value < met.baseline_value * 0.95
                else "stable"
            ),
        )
        for met in patient.metrics
    ]

    wearable_baselines = {}
    for met in patient.metrics:
        if met.metric == "tremor_amplitude":
            wearable_baselines["tremor_amplitude_baseline"] = met.baseline_value
        if met.metric == "gait_speed_m_s":
            wearable_baselines["gait_speed_baseline"] = met.baseline_value

    state = PatientState(
        patient_id=patient.id,
        demographics=Demographics(age=patient.age, sex=patient.sex),
        disease_state=DiseaseState(conditions=conditions),
        symptom_state=SymptomState(),
        medication_state=MedicationState(
            active_medications=active_meds,
            allergies=[a.substance for a in patient.allergies],
            adherence_pct_30d=adherence_avg,
        ),
        risk_state=RiskState(),
        behavioral_state=BehavioralState(),
        cognitive_state=CognitiveState(),
        adherence_state=AdherenceState(),
        environmental_state=EnvironmentalState(),
        longitudinal_trends=trends,
        wearable_metrics=wearable_baselines,
        patient_preferences=PatientPreferences(preferred_language=patient.preferred_language, notification_opt_in=patient.notification_opt_in),
        clinician_constraints=ClinicianConstraints(restricted_actions=patient.clinician_restricted_actions or []),
    )
    state.data_completeness = state.compute_completeness()
    return state


def persist_workflow(db: Session, state: WorkflowState) -> m.Workflow:
    wf = m.Workflow(
        id=state.workflow_id,
        patient_id=state.patient_id,
        trigger_reason=state.trigger_reason,
        status=state.status,
        is_emergency=state.is_emergency,
        started_at=datetime.fromisoformat(state.started_at),
        completed_at=datetime.fromisoformat(state.completed_at) if state.completed_at else None,
        result_json={
            "consensus": {
                "top_hypothesis": state.consensus.top_hypothesis if state.consensus else None,
                "disagreement": state.consensus.disagreement if state.consensus else None,
                "hypotheses": [h.__dict__ for h in state.consensus.hypotheses] if state.consensus else [],
                "action_permitted": state.consensus.action_permitted if state.consensus else None,
                "reason": state.consensus.reason if state.consensus else None,
            } if state.consensus else None,
        },
    )
    db.add(wf)

    for agent_name, result in state.agent_results.items():
        db.add(m.AgentEventLog(
            workflow_id=state.workflow_id,
            patient_id=state.patient_id,
            agent_name=agent_name,
            status=result.status,
            confidence=result.confidence,
            result_json=result.model_dump(),
        ))

    for event in state.audit_events:
        db.add(m.AuditEvent(
            workflow_id=state.workflow_id,
            patient_id=state.patient_id,
            event=event["event"],
            detail_json=event["detail"],
            timestamp=datetime.fromisoformat(event["timestamp"]),
        ))

    for action in state.proposed_actions:
        db.add(m.ActionRecord(
            id=action.action_id,
            patient_id=action.patient_id,
            workflow_id=action.workflow_id,
            action_type=action.action_type,
            reason=action.reason,
            evidence_json=action.evidence,
            risk_level=action.risk_level,
            autonomy_level=action.autonomy_level,
            requires_human_approval=action.requires_human_approval,
            approved_by=action.approved_by,
            status=action.status,
            safety_reasons_json=action.safety_reasons,
        ))
        # Medium+ risk / human-review actions surface as Alerts for the UI.
        if action.requires_human_approval or action.risk_level in {"high", "restricted"}:
            db.add(m.Alert(
                patient_id=action.patient_id,
                workflow_id=action.workflow_id,
                severity="critical" if state.is_emergency and action.action_type == "emergency_escalation" else (
                    "high" if action.risk_level in {"high", "restricted"} else "medium"
                ),
                reason=action.reason,
                evidence_json=action.evidence,
                status="open",
            ))

    db.commit()
    db.refresh(wf)
    return wf
