from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models as m
from app.db.session import get_db
from app.orchestrator.orchestrator import run_workflow
from app.schemas.api import WorkflowTriggerIn
from app.services.auth import get_current_user
from app.services.repository import build_patient_state, persist_workflow

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("/neurological-assessment/{patient_id}")
async def trigger_neurological_assessment(
    patient_id: str,
    payload: WorkflowTriggerIn,
    db: Session = Depends(get_db),
    user: m.User = Depends(get_current_user),
):
    patient = db.get(m.Patient, patient_id)
    if not patient:
        raise HTTPException(404, "patient not found")

    patient_state = build_patient_state(db, patient)

    input_payload = {
        "symptoms": [s.model_dump() for s in payload.observations.symptoms],
        "medication_logs": [ml.model_dump() for ml in payload.observations.medication_logs],
        "wearable": payload.observations.wearable,
        "vitals": payload.observations.vitals,
        "caregiver_notes": payload.observations.caregiver_notes,
    }

    workflow_state = await run_workflow(
        patient_id=patient_id,
        trigger_reason=payload.trigger_reason,
        input_payload=input_payload,
        patient_state=patient_state,
    )

    # Persist raw observations too, tagged with this workflow id.
    for s in payload.observations.symptoms:
        db.add(m.Observation(patient_id=patient_id, workflow_id=workflow_state.workflow_id, modality="symptom", payload=s.model_dump()))
    if payload.observations.caregiver_notes:
        db.add(m.Observation(patient_id=patient_id, workflow_id=workflow_state.workflow_id, modality="caregiver_report",
                              payload={"note": payload.observations.caregiver_notes}))

    persist_workflow(db, workflow_state)

    return _serialize_workflow(workflow_state)


@router.get("/{workflow_id}")
def get_workflow(workflow_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    wf = db.get(m.Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "workflow not found")
    events = db.query(m.AgentEventLog).filter(m.AgentEventLog.workflow_id == workflow_id).order_by(m.AgentEventLog.timestamp).all()
    actions = db.query(m.ActionRecord).filter(m.ActionRecord.workflow_id == workflow_id).all()
    audit = db.query(m.AuditEvent).filter(m.AuditEvent.workflow_id == workflow_id).order_by(m.AuditEvent.timestamp).all()
    return {
        "workflow_id": wf.id,
        "patient_id": wf.patient_id,
        "status": wf.status,
        "is_emergency": wf.is_emergency,
        "trigger_reason": wf.trigger_reason,
        "started_at": wf.started_at.isoformat(),
        "completed_at": wf.completed_at.isoformat() if wf.completed_at else None,
        "consensus": wf.result_json.get("consensus"),
        "agent_events": [{"agent_name": e.agent_name, "status": e.status, "confidence": e.confidence, "result": e.result_json} for e in events],
        "actions": [{
            "action_id": a.id, "action_type": a.action_type, "status": a.status, "risk_level": a.risk_level,
            "autonomy_level": a.autonomy_level, "requires_human_approval": a.requires_human_approval, "reason": a.reason,
        } for a in actions],
        "audit_trail": [{"timestamp": e.timestamp.isoformat(), "event": e.event, "detail": e.detail_json} for e in audit],
    }


def _serialize_workflow(state) -> dict:
    return {
        "workflow_id": state.workflow_id,
        "patient_id": state.patient_id,
        "status": state.status,
        "is_emergency": state.is_emergency,
        "trigger_reason": state.trigger_reason,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "consensus": {
            "top_hypothesis": state.consensus.top_hypothesis,
            "disagreement": state.consensus.disagreement,
            "hypotheses": [h.__dict__ for h in state.consensus.hypotheses],
            "action_permitted": state.consensus.action_permitted,
            "reason": state.consensus.reason,
        } if state.consensus else None,
        "agent_results": {name: r.model_dump() for name, r in state.agent_results.items()},
        "proposed_actions": [a.__dict__ for a in state.proposed_actions],
        "audit_trail": state.audit_events,
        "messages": [msg.model_dump() for msg in state.messages],
    }
