from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.supabase_client import get_authed_client
from app.db import models as m
from app.db.session import get_db
from app.schemas.api import MedicationIn, ObservationBatchIn, PatientCreate, PatientOut
from app.services.auth import get_current_user

router = APIRouter(prefix="/patients", tags=["patients"])

# -- MIGRATED to Supabase (real RLS, real auth) -----------------------------


@router.post("", response_model=PatientOut)
def create_patient(payload: PatientCreate, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])

    profile = client.table("profiles").select("role").eq("id", user["sub"]).execute()
    caller_role = profile.data[0]["role"] if profile.data else "patient"

    row = {
        "full_name": payload.full_name,
        "age": payload.age,
        "sex": payload.sex,
        "preferred_language": payload.preferred_language,
        "user_id": user["sub"] if caller_role == "patient" else None,
        "clinician_id": payload.clinician_id or (user["sub"] if caller_role == "clinician" else None),
    }
    result = client.table("patients").insert(row).execute()
    if not result.data:
        raise HTTPException(500, "Failed to create patient")
    p = result.data[0]
    # NOTE: condition tracking (payload.conditions) is not persisted yet --
    # the old Condition table lived on SQLite tied to the old local Patient
    # table, which no longer applies. Migrating conditions/observations/
    # medications to Supabase is upcoming, tracked in docs/ROADMAP.md.
    return PatientOut(id=p["id"], full_name=p["full_name"], age=p["age"], sex=p["sex"], conditions=[])


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    result = client.table("patients").select("*").eq("id", patient_id).execute()
    if not result.data:
        raise HTTPException(404, "patient not found")
    p = result.data[0]
    return PatientOut(id=p["id"], full_name=p["full_name"], age=p["age"], sex=p["sex"], conditions=[])


@router.get("")
def list_patients(user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    result = client.table("patients").select("*").execute()
    return [
        PatientOut(id=p["id"], full_name=p["full_name"], age=p["age"], sex=p["sex"], conditions=[])
        for p in result.data
    ]


# -- NOT YET MIGRATED -- still SQLite, will 404 for any patient created ----
# -- above via Supabase (their id won't exist in the local DB). This is a --
# -- known, deliberate gap -- Observation/Medication/Workflow/etc. move to -
# -- Supabase in a later step, table by table, same as patients above. ----


@router.post("/{patient_id}/observations")
def add_observations(patient_id: str, payload: ObservationBatchIn, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
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
def add_medication(patient_id: str, payload: MedicationIn, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    patient = db.get(m.Patient, patient_id)
    if not patient:
        raise HTTPException(404, "patient not found")
    med = m.Medication(patient_id=patient_id, **payload.model_dump())
    db.add(med)
    db.commit()
    return {"id": med.id}


@router.get("/{patient_id}/timeline")
def get_timeline(patient_id: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    events = db.query(m.AuditEvent).filter(m.AuditEvent.patient_id == patient_id).order_by(m.AuditEvent.timestamp).all()
    return [{"timestamp": e.timestamp.isoformat(), "event": e.event, "detail": e.detail_json} for e in events]


@router.get("/{patient_id}/risk")
def get_risk(patient_id: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
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
def get_agent_events(patient_id: str, workflow_id: str | None = None, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    q = db.query(m.AgentEventLog).filter(m.AgentEventLog.patient_id == patient_id)
    if workflow_id:
        q = q.filter(m.AgentEventLog.workflow_id == workflow_id)
    events = q.order_by(m.AgentEventLog.timestamp).all()
    return [{
        "workflow_id": e.workflow_id, "agent_name": e.agent_name, "status": e.status,
        "confidence": e.confidence, "timestamp": e.timestamp.isoformat(), "result": e.result_json,
    } for e in events]


@router.get("/{patient_id}/explanations")
def get_explanations(patient_id: str, workflow_id: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
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
def get_patient_alerts(patient_id: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    alerts = db.query(m.Alert).filter(m.Alert.patient_id == patient_id).order_by(m.Alert.created_at.desc()).all()
    return [{
        "id": a.id, "severity": a.severity, "reason": a.reason, "evidence": a.evidence_json,
        "status": a.status, "created_at": a.created_at.isoformat(),
    } for a in alerts]