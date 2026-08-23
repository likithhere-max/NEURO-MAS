from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models as m
from app.db.session import get_db
from app.schemas.api import MedicationIn, ObservationBatchIn, PatientCreate, PatientOut
from app.services.auth import get_current_user
from app.services.knowledge_base import get_knowledge_base

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("", response_model=PatientOut)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    kb = get_knowledge_base()
    patient = m.Patient(
        full_name=payload.full_name,
        age=payload.age,
        sex=payload.sex,
        preferred_language=payload.preferred_language,
        user_id=user.id if user.role == "patient" else None,
        clinician_id=payload.clinician_id or (user.id if user.role == "clinician" else None),
    )
    db.add(patient)
    db.flush()

    for cond_ref in payload.conditions:
        record = kb.get(cond_ref) or next((d for d in kb.list_all() if d["preferred_name"] == cond_ref), None)
        db.add(m.Condition(
            patient_id=patient.id,
            disease_id=record["disease_id"] if record else cond_ref,
            preferred_name=record["preferred_name"] if record else cond_ref,
        ))
    db.commit()
    db.refresh(patient)
    return PatientOut(id=patient.id, full_name=patient.full_name, age=patient.age, sex=patient.sex,
                       conditions=[c.preferred_name for c in patient.conditions])


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    patient = db.get(m.Patient, patient_id)
    if not patient:
        raise HTTPException(404, "patient not found")
    return PatientOut(id=patient.id, full_name=patient.full_name, age=patient.age, sex=patient.sex,
                       conditions=[c.preferred_name for c in patient.conditions])


@router.get("")
def list_patients(db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    patients = db.query(m.Patient).all()
    return [PatientOut(id=p.id, full_name=p.full_name, age=p.age, sex=p.sex,
                        conditions=[c.preferred_name for c in p.conditions]) for p in patients]


@router.post("/{patient_id}/observations")
def add_observations(patient_id: str, payload: ObservationBatchIn, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    patient = db.get(m.Patient, patient_id)
    if not patient:
        raise HTTPException(404, "patient not found")
    count = 0
    for s in payload.symptoms:
        db.add(m.Observation(patient_id=patient_id, modality="symptom", payload=s.model_dump()))
        count += 1
    for med in payload.medication_logs:
        db.add(m.Observation(patient_id=patient_id, modality="medication_log", payload=med.model_dump()))
        count += 1
    if payload.wearable:
        db.add(m.Observation(patient_id=patient_id, modality="wearable", payload=payload.wearable))
        count += 1
    if payload.vitals:
        db.add(m.Observation(patient_id=patient_id, modality="vital", payload=payload.vitals))
        count += 1
    if payload.caregiver_notes:
        db.add(m.Observation(patient_id=patient_id, modality="caregiver_report", payload={"note": payload.caregiver_notes}))
        count += 1
    db.commit()
    return {"stored": count}


@router.post("/{patient_id}/medications")
def add_medication(patient_id: str, payload: MedicationIn, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    patient = db.get(m.Patient, patient_id)
    if not patient:
        raise HTTPException(404, "patient not found")
    med = m.Medication(patient_id=patient_id, **payload.model_dump())
    db.add(med)
    db.commit()
    return {"id": med.id}


@router.get("/{patient_id}/timeline")
def get_timeline(patient_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    events = db.query(m.AuditEvent).filter(m.AuditEvent.patient_id == patient_id).order_by(m.AuditEvent.timestamp).all()
    return [{"timestamp": e.timestamp.isoformat(), "event": e.event, "detail": e.detail_json} for e in events]


@router.get("/{patient_id}/risk")
def get_risk(patient_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    latest = (
        db.query(m.AgentEventLog)
        .filter(m.AgentEventLog.patient_id == patient_id, m.AgentEventLog.agent_name == "risk_stratification_agent")
        .order_by(m.AgentEventLog.timestamp.desc())
        .first()
    )
    if not latest:
        return {"message": "no risk assessment yet for this patient"}
    return latest.result_json.get("raw_payload", {})


@router.get("/{patient_id}/agent-events")
def get_agent_events(patient_id: str, workflow_id: str | None = None, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    q = db.query(m.AgentEventLog).filter(m.AgentEventLog.patient_id == patient_id)
    if workflow_id:
        q = q.filter(m.AgentEventLog.workflow_id == workflow_id)
    events = q.order_by(m.AgentEventLog.timestamp).all()
    return [{
        "workflow_id": e.workflow_id, "agent_name": e.agent_name, "status": e.status,
        "confidence": e.confidence, "timestamp": e.timestamp.isoformat(), "result": e.result_json,
    } for e in events]


@router.get("/{patient_id}/explanations")
def get_explanations(patient_id: str, workflow_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    wf = db.get(m.Workflow, workflow_id)
    if not wf or wf.patient_id != patient_id:
        raise HTTPException(404, "workflow not found")
    events = db.query(m.AgentEventLog).filter(m.AgentEventLog.workflow_id == workflow_id).all()
    actions = db.query(m.ActionRecord).filter(m.ActionRecord.workflow_id == workflow_id).all()
    return {
        "what_happened": wf.trigger_reason,
        "consensus": wf.result_json.get("consensus"),
        "which_agents_participated": [e.agent_name for e in events],
        "what_each_agent_found": {
            e.agent_name: {
                "confidence": e.confidence,
                "reasoning_summary": e.result_json.get("reasoning_summary"),
                "findings": e.result_json.get("findings"),
            } for e in events
        },
        "safety_and_actions": [
            {
                "action_type": a.action_type, "status": a.status, "risk_level": a.risk_level,
                "autonomy_level": a.autonomy_level, "requires_human_approval": a.requires_human_approval,
                "safety_reasons": a.safety_reasons_json,
            } for a in actions
        ],
    }


@router.get("/{patient_id}/alerts")
def get_patient_alerts(patient_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    alerts = db.query(m.Alert).filter(m.Alert.patient_id == patient_id).order_by(m.Alert.created_at.desc()).all()
    return [{
        "id": a.id, "severity": a.severity, "reason": a.reason, "evidence": a.evidence_json,
        "status": a.status, "created_at": a.created_at.isoformat(),
    } for a in alerts]
