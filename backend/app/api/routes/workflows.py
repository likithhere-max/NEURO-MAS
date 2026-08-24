from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.supabase_client import get_authed_client
from app.orchestrator.orchestrator import run_workflow
from app.schemas.api import WorkflowTriggerIn
from app.services.auth import get_current_user
from app.services.repository import build_patient_state, persist_workflow

router = APIRouter(prefix="/workflows", tags=["workflows"])


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
    persist_workflow(client, workflow_state)

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