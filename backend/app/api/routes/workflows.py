from __future__ import annotations

import asyncio
import logging
import re
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException

from app.core import privacy
from app.core.supabase_client import get_authed_client
from app.orchestrator.orchestrator import run_workflow
from app.schemas.api import WorkflowTriggerIn
from app.schemas.symptom_assessment import SymptomAssessmentIn
from app.services.auth import get_current_user
from app.services import notification_service
from app.services.repository import (
    build_patient_state,
    consensus_to_dict,
    persist_workflow,
    record_emergency_checkpoint,
)
from app.services.symptom_extraction import (
    SymptomExtractionError,
    extract_symptom_profile,
    extracted_profile_to_symptom_ins,
)

from app.core.config import get_settings
from app.core.ratelimit import rate_limit_user  # noqa: E402

router = APIRouter(prefix="/workflows", tags=["workflows"])
logger = logging.getLogger("neuromas.workflows")


_IDEM_RE = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


def _workflow_id_for(patient_id: str, idempotency_key: str | None) -> str | None:
    """Deterministic workflow id from (patient, Idempotency-Key): a retried request maps to the SAME row, and the
    primary key makes a second concurrent insert fail instead of duplicating the run."""
    if idempotency_key is None:
        return None
    if not _IDEM_RE.match(idempotency_key):
        raise HTTPException(422, "Idempotency-Key must be 8-128 characters from [A-Za-z0-9._:-]")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"neuromas:{patient_id}:{idempotency_key}"))


def _stored_complete_workflow(client, workflow_id: str | None):
    if not workflow_id:
        return None
    rows = client.table("workflows").select("*").eq("id", workflow_id).execute().data
    if rows and (rows[0].get("result_json") or {}).get("persist_state", "complete") == "complete":
        return rows[0]
    return None


async def _deliver_notifications(client, workflow_state, patient: dict) -> None:
    """After the workflow is stored, hand permitted actions to the configured channel (if any). A failure here is
    logged and escalated by the dispatcher; it never changes the already-stored assessment or the HTTP result."""
    try:
        # Admin credentials are requested only when a delivery channel is configured (default: none).
        admin = None
        if (get_settings().notification_channel or "none").lower() != "none":
            from app.core.supabase_client import get_supabase_admin_client
            admin = get_supabase_admin_client()
        dispatcher = notification_service.build_dispatcher(
            admin, escalate=notification_service.make_alert_escalation(client),
            audit=lambda e: logger.info("notification_audit %s", e))
        if dispatcher is None:
            return
        await notification_service.dispatch_workflow_notifications(workflow_state, patient, dispatcher, client)
    except Exception as exc:
        logger.error("notification_stage_failed error_class=%s", type(exc).__name__)


def _emergency_checkpoint(client):
    async def cb(state, name):
        # blocking Supabase call off the event loop; failure is logged by the orchestrator, never fatal
        await asyncio.to_thread(record_emergency_checkpoint, client, state)
    return cb


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


@router.post("/neurological-assessment/{patient_id}",
             dependencies=[Depends(rate_limit_user("workflow", "workflow_rate_limit_per_minute"))])
async def trigger_neurological_assessment(
    patient_id: str,
    payload: WorkflowTriggerIn,
    user: dict = Depends(get_current_user),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    client = get_authed_client(user["_access_token"])
    workflow_id = _workflow_id_for(patient_id, idempotency_key)

    patient_result = client.table("patients").select("*").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    patient = patient_result.data[0]

    if _stored_complete_workflow(client, workflow_id):
        return {"idempotent_replay": True, **_load_workflow(client, workflow_id)}

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
        workflow_id=workflow_id,
        on_checkpoint=_emergency_checkpoint(client),
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

    await _deliver_notifications(client, workflow_state, patient)
    return _serialize_workflow(workflow_state)


@router.post("/symptom-assessment/{patient_id}",
             dependencies=[Depends(rate_limit_user("workflow", "workflow_rate_limit_per_minute"))])
async def trigger_symptom_assessment(
    patient_id: str,
    payload: SymptomAssessmentIn,
    user: dict = Depends(get_current_user),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
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
    workflow_id = _workflow_id_for(patient_id, idempotency_key)

    patient_result = client.table("patients").select("*").eq("id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(404, "patient not found")
    patient = patient_result.data[0]

    if not payload.free_text or not payload.free_text.strip():
        raise HTTPException(422, "free_text is required")

    if _stored_complete_workflow(client, workflow_id):      # replay BEFORE paying for another LLM extraction
        stored = client.table("symptom_assessments").select("*").eq("workflow_id", workflow_id).limit(1).execute().data
        return {"idempotent_replay": True, "assessment": stored[0] if stored else None,
                "extracted": stored[0]["extracted_json"] if stored else None,
                "workflow": _load_workflow(client, workflow_id)}

    with privacy.processing_scope(purpose="symptom_extraction", consent=bool(patient.get("external_ai_consent", False)),
                                  identifiers=[patient.get("full_name")]):
        gate = privacy.decide("generate")
        if not gate.allowed:
            # Extraction has no local fallback, so say why instead of returning a misleading 502.
            raise HTTPException(403, f"external AI processing is not permitted for this patient ({gate.reason})")
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
        workflow_id=workflow_id,
        on_checkpoint=_emergency_checkpoint(client),
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

    await _deliver_notifications(client, workflow_state, patient)
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
    return _load_workflow(client, workflow_id)


def _load_workflow(client, workflow_id: str) -> dict:
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
        "timed_out_agents": wf["result_json"].get("timed_out_agents", []),
        "deadline_exceeded": wf["result_json"].get("deadline_exceeded", False),
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
        "timed_out_agents": getattr(state, "timed_out_agents", []),
        "deadline_exceeded": getattr(state, "deadline_exceeded", False),
        "emergency_screen": getattr(state, "emergency_screen", {}),
        "workflow_id": state.workflow_id,
        "patient_id": state.patient_id,
        "status": state.status,
        "is_emergency": state.is_emergency,
        "trigger_reason": state.trigger_reason,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "consensus": consensus_to_dict(state.consensus),
        "agent_results": {name: r.model_dump() for name, r in state.agent_results.items()},
        "proposed_actions": [a.__dict__ for a in state.proposed_actions],
        "audit_trail": state.audit_events,
        "messages": [msg.model_dump() for msg in state.messages],
    }