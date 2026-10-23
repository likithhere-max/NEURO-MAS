from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.supabase_client import get_authed_client
from app.orchestrator.orchestrator import run_workflow
from app.schemas.api import WorkflowTriggerIn
from app.schemas.symptom_assessment import SymptomAssessmentIn
from app.services.auth import get_current_user
from app.services.repository import build_patient_state, persist_workflow
from app.services.symptom_extraction import (
    SymptomExtractionError,
    extract_symptom_profile,
    extracted_profile_to_symptom_ins,
)

router = APIRouter(prefix="/workflows", tags=["workflows"])
logger = logging.getLogger("neuromas.workflows")


def _persist_or_degrade(client, workflow_state) -> bool:
    """Persist the workflow. Audit fix (High): persistence is 5+ non-transactional inserts; a failure
    used to turn into a bare 500 and the computed emergency flag never reached the caller.

    Emergency runs: swallow the persistence error (logged), return the result with persisted=False so
    the patient is still told. Non-emergency runs: re-raise (nothing safety-critical to salvage).
    Returns True if persisted."""
    try:
        persist_workflow(client, workflow_state)
        return True
    except Exception as exc:
        logger.error("persist_workflow_failed workflow_id=%s emergency=%s error=%r",
                     workflow_state.workflow_id, workflow_state.is_emergency, exc)
        if workflow_state.is_emergency:
            return False
        raise HTTPException(500, "could not save the assessment; please retry")


@router.post("/neurological-assessment/{patient_id}")
async def trigger_neurological_assessment(
    patient_id: str,
    payload: WorkflowTriggerIn,
    user: dict = Depends(get_current_user),
):
    client = get_authed_client(user["_access_token"])

    patient_result = client.table("patients").select("*").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    patient = patient_result.data[0]

    patient_state = build_patient_state(client, patient)
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

    # persist_workflow() must run BEFORE the observations insert below: unlike
    # the old SQLAlchemy session (which auto-ordered inserts by foreign-key
    # dependency within one flush/commit), each Supabase REST call is its own
    # immediate insert -- and observations.workflow_id has a foreign key to
    # workflows.id, so the workflows row has to exist first.
    persisted = _persist_or_degrade(client, workflow_state)
    if not persisted:
        return _serialize_workflow(workflow_state, persisted=False)

    # Persist raw observations too, tagged with this workflow id.
    observation_rows = [
        {
            "patient_id": patient_id,
            "workflow_id": workflow_state.workflow_id,
            "modality": "symptom",
            "payload": s.model_dump(),
        }
        for s in payload.observations.symptoms
    ]
    if payload.observations.caregiver_notes:
        observation_rows.append({
            "patient_id": patient_id,
            "workflow_id": workflow_state.workflow_id,
            "modality": "caregiver_report",
            "payload": {"note": payload.observations.caregiver_notes},
        })
    if observation_rows:
        client.table("observations").insert(observation_rows).execute()

    return _serialize_workflow(workflow_state)


@router.post("/symptom-assessment/{patient_id}")
async def trigger_symptom_assessment(
    patient_id: str,
    payload: SymptomAssessmentIn,
    user: dict = Depends(get_current_user),
):
    """
    Real natural-language symptom entry point (2-day review sprint feature).

    Extracts structured clinical features from the patient's own free-text
    description, converts them into the same SymptomIn contract the manual
    /workflows/neurological-assessment/{id} endpoint already uses, and runs
    the UNMODIFIED existing multi-agent pipeline (run_workflow) over them.
    This endpoint adds a real input layer -- it does not add a second
    diagnostic pipeline.
    """
    client = get_authed_client(user["_access_token"])

    patient_result = client.table("patients").select("*").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    patient = patient_result.data[0]

    if not payload.free_text or not payload.free_text.strip():
        raise HTTPException(422, "free_text is required")

    try:
        extracted = await extract_symptom_profile(payload)
    except SymptomExtractionError as exc:
        raise HTTPException(502, str(exc))

    symptom_ins = extracted_profile_to_symptom_ins(payload.free_text, extracted)

    patient_state = build_patient_state(client, patient)
    input_payload = {
        "symptoms": [s.model_dump() for s in symptom_ins],
        "medication_logs": [],
        "wearable": {},
        "vitals": {},
        "caregiver_notes": None,
    }
    workflow_state = await run_workflow(
        patient_id=patient_id,
        trigger_reason="patient_symptom_assessment",
        input_payload=input_payload,
        patient_state=patient_state,
    )

    # Same ordering requirement as trigger_neurological_assessment above:
    # persist_workflow() must run before the observations insert, since
    # observations.workflow_id has a foreign key to workflows.id.
    persisted = _persist_or_degrade(client, workflow_state)
    if not persisted:
        return {"assessment": None, "extracted": extracted.model_dump(), "workflow": _serialize_workflow(workflow_state, persisted=False)}

    observation_rows = [
        {
            "patient_id": patient_id,
            "workflow_id": workflow_state.workflow_id,
            "modality": "symptom",
            "payload": s.model_dump(),
        }
        for s in symptom_ins
    ]
    if observation_rows:
        client.table("observations").insert(observation_rows).execute()

    assessment_row = {
        "patient_id": patient_id,
        "workflow_id": workflow_state.workflow_id,
        "free_text": payload.free_text,
        "extracted_json": extracted.model_dump(),
    }
    assessment_result = client.table("symptom_assessments").insert(assessment_row).execute()
    assessment = assessment_result.data[0] if assessment_result.data else None

    return {
        "assessment": assessment,
        "extracted": extracted.model_dump(),
        "workflow": _serialize_workflow(workflow_state),
    }


@router.get("/symptom-assessment/{patient_id}/history")
def list_symptom_assessments(patient_id: str, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    result = (
        client.table("symptom_assessments")
        .select("*")
        .eq("patient_id", patient_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


@router.get("/{workflow_id}")
def get_workflow(workflow_id: str, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])

    wf_result = client.table("workflows").select("*").eq("id", workflow_id).execute()
    if not wf_result.data:
        raise HTTPException(404, "workflow not found")
    wf = wf_result.data[0]

    events = client.table("agent_event_log").select("*").eq("workflow_id", workflow_id).order("timestamp").execute().data
    actions = client.table("actions").select("*").eq("workflow_id", workflow_id).execute().data
    audit = client.table("audit_events").select("*").eq("workflow_id", workflow_id).order("timestamp").execute().data

    return {
        "workflow_id": wf["id"],
        "patient_id": wf["patient_id"],
        "status": wf["status"],
        "is_emergency": wf["is_emergency"],
        "trigger_reason": wf["trigger_reason"],
        "started_at": wf["started_at"],
        "completed_at": wf["completed_at"],
        "consensus": wf["result_json"].get("consensus"),
        "degraded": wf["result_json"].get("degraded", False),
        "failed_agents": wf["result_json"].get("failed_agents", []),
        "agent_events": [
            {"agent_name": e["agent_name"], "status": e["status"], "confidence": e["confidence"], "result": e["result_json"]}
            for e in events
        ],
        "actions": [{
            "action_id": a["id"], "action_type": a["action_type"], "status": a["status"], "risk_level": a["risk_level"],
            "autonomy_level": a["autonomy_level"], "requires_human_approval": a["requires_human_approval"], "reason": a["reason"],
        } for a in actions],
        "audit_trail": [{"timestamp": e["timestamp"], "event": e["event"], "detail": e["detail_json"]} for e in audit],
    }


def _serialize_workflow(state, persisted: bool = True) -> dict:
    return {
        "persisted": persisted,
        "degraded": getattr(state, "degraded", False),
        "failed_agents": getattr(state, "failed_agents", []),
        "emergency_screen": getattr(state, "emergency_screen", {}),
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