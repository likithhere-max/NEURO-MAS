from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.supabase_client import get_authed_client
from app.schemas.api import (
    AllergyIn,
    ConditionIn,
    LongitudinalMetricIn,
    MedicationIn,
    ObservationBatchIn,
    PatientCreate,
    PatientOut,
)
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
    # NOTE: condition tracking (payload.conditions) is still not persisted here --
    # each string is ambiguously either a disease_id or a preferred_name (see the
    # field's own docstring), and resolving that needs a real knowledge-base
    # lookup decision that hasn't been made yet. Use POST /patients/{id}/conditions
    # below for now, which takes explicit, unambiguous fields.
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


@router.post("/{patient_id}/conditions")
def add_condition(patient_id: str, payload: ConditionIn, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    patient_result = client.table("patients").select("id").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    row = {"patient_id": patient_id, **payload.model_dump()}
    result = client.table("conditions").insert(row).execute()
    if not result.data:
        raise HTTPException(500, "Failed to add condition")
    return result.data[0]


@router.post("/{patient_id}/allergies")
def add_allergy(patient_id: str, payload: AllergyIn, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    patient_result = client.table("patients").select("id").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    row = {"patient_id": patient_id, **payload.model_dump()}
    result = client.table("allergies").insert(row).execute()
    if not result.data:
        raise HTTPException(500, "Failed to add allergy")
    return result.data[0]


@router.post("/{patient_id}/longitudinal-metrics")
def add_longitudinal_metric(patient_id: str, payload: LongitudinalMetricIn, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    patient_result = client.table("patients").select("id").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    row = {"patient_id": patient_id, **payload.model_dump()}
    result = client.table("longitudinal_metrics").insert(row).execute()
    if not result.data:
        raise HTTPException(500, "Failed to add longitudinal metric")
    return result.data[0]


@router.post("/{patient_id}/observations")
def add_observations(patient_id: str, payload: ObservationBatchIn, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    patient_result = client.table("patients").select("id").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")

    rows = []
    for s in payload.symptoms:
        rows.append({"patient_id": patient_id, "modality": "symptom", "payload": s.model_dump()})
    for med in payload.medication_logs:
        rows.append({"patient_id": patient_id, "modality": "medication_log", "payload": med.model_dump()})
    if payload.wearable:
        rows.append({"patient_id": patient_id, "modality": "wearable", "payload": payload.wearable})
    if payload.vitals:
        rows.append({"patient_id": patient_id, "modality": "vital", "payload": payload.vitals})
    if payload.caregiver_notes:
        rows.append({"patient_id": patient_id, "modality": "caregiver_report", "payload": {"note": payload.caregiver_notes}})
    if rows:
        client.table("observations").insert(rows).execute()
    return {"stored": len(rows)}


@router.post("/{patient_id}/medications")
def add_medication(patient_id: str, payload: MedicationIn, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    patient_result = client.table("patients").select("id").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    row = {"patient_id": patient_id, **payload.model_dump()}
    result = client.table("medications").insert(row).execute()
    if not result.data:
        raise HTTPException(500, "Failed to add medication")
    return {"id": result.data[0]["id"]}


@router.get("/{patient_id}/timeline")
def get_timeline(patient_id: str, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    events = client.table("audit_events").select("*").eq("patient_id", patient_id).order("timestamp").execute().data
    return [{"timestamp": e["timestamp"], "event": e["event"], "detail": e["detail_json"]} for e in events]


@router.get("/{patient_id}/risk")
def get_risk(patient_id: str, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    result = (
        client.table("agent_event_log")
        .select("*")
        .eq("patient_id", patient_id)
        .eq("agent_name", "risk_stratification_agent")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return {"message": "no risk assessment yet for this patient"}
    return result.data[0]["result_json"].get("raw_payload", {})


@router.get("/{patient_id}/agent-events")
def get_agent_events(patient_id: str, workflow_id: str | None = None, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    query = client.table("agent_event_log").select("*").eq("patient_id", patient_id)
    if workflow_id:
        query = query.eq("workflow_id", workflow_id)
    events = query.order("timestamp").execute().data
    return [{
        "workflow_id": e["workflow_id"], "agent_name": e["agent_name"], "status": e["status"],
        "confidence": e["confidence"], "timestamp": e["timestamp"], "result": e["result_json"],
    } for e in events]


@router.get("/{patient_id}/explanations")
def get_explanations(patient_id: str, workflow_id: str, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    wf_result = client.table("workflows").select("*").eq("id", workflow_id).execute()
    if not wf_result.data or wf_result.data[0]["patient_id"] != patient_id:
        raise HTTPException(404, "workflow not found")
    wf = wf_result.data[0]
    events = client.table("agent_event_log").select("*").eq("workflow_id", workflow_id).execute().data
    actions = client.table("actions").select("*").eq("workflow_id", workflow_id).execute().data
    return {
        "what_happened": wf["trigger_reason"],
        "consensus": wf["result_json"].get("consensus"),
        "which_agents_participated": [e["agent_name"] for e in events],
        "what_each_agent_found": {
            e["agent_name"]: {
                "confidence": e["confidence"],
                "reasoning_summary": e["result_json"].get("reasoning_summary"),
                "findings": e["result_json"].get("findings"),
            } for e in events
        },
        "safety_and_actions": [
            {
                "action_type": a["action_type"], "status": a["status"], "risk_level": a["risk_level"],
                "autonomy_level": a["autonomy_level"], "requires_human_approval": a["requires_human_approval"],
                "safety_reasons": a["safety_reasons_json"],
            } for a in actions
        ],
    }


@router.get("/{patient_id}/alerts")
def get_patient_alerts(patient_id: str, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    alerts = client.table("alerts").select("*").eq("patient_id", patient_id).order("created_at", desc=True).execute().data
    return [{
        "id": a["id"], "severity": a["severity"], "reason": a["reason"], "evidence": a["evidence_json"],
        "status": a["status"], "created_at": a["created_at"],
    } for a in alerts]