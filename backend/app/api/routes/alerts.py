from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.core.supabase_client import get_authed_client
from app.schemas.api import AlertAcknowledgeIn
from app.services.auth import get_current_user

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _caller_role(client, user: dict) -> str:
    profile = client.table("profiles").select("role").eq("id", user["sub"]).execute()
    return profile.data[0]["role"] if profile.data else "patient"


@router.post("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, payload: AlertAcknowledgeIn, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    # Audit fix (High): any authenticated role (including the patient the alert is about) could
    # acknowledge/dismiss a critical alert. Acknowledgement is a clinical-staff action.
    if _caller_role(client, user) != "clinician":
        raise HTTPException(403, "only a clinician may acknowledge an alert")
    result = client.table("alerts").update({
        "status": "acknowledged",
        "acknowledged_by": user["sub"],
        "acknowledged_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", alert_id).execute()
    if not result.data:
        raise HTTPException(404, "alert not found")
    alert = result.data[0]
    return {"id": alert["id"], "status": alert["status"]}


@router.get("")
def list_alerts(status_filter: str | None = None, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    query = client.table("alerts").select("*")
    if status_filter:
        query = query.eq("status", status_filter)
    alerts = query.order("created_at", desc=True).execute().data
    return [{
        "id": a["id"], "patient_id": a["patient_id"], "severity": a["severity"], "reason": a["reason"],
        "status": a["status"], "created_at": a["created_at"],
    } for a in alerts]


@router.post("/actions/{action_id}/approve")
def approve_action(action_id: str, user: dict = Depends(get_current_user)):
    client = get_authed_client(user["_access_token"])
    if _caller_role(client, user) != "clinician":
        raise HTTPException(403, "only a clinician may authorize a restricted action")
    # Only an action that is actually awaiting approval can be approved (previously ANY visible action
    # could be flipped to "executed", including already-approved or auto-proposed ones).
    # NOTE: the status value "executed" is kept for DB/UI compatibility, but here it means "approved by a
    # clinician", not "delivered" -- see docs/audit/OUTSTANDING_ISSUES.md (introduce an "approved" status).
    result = (
        client.table("actions")
        .update({"status": "executed", "approved_by": user["sub"]})
        .eq("id", action_id)
        .eq("status", "pending_approval")
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "no pending action with that id")
    action = result.data[0]
    return {"id": action["id"], "status": action["status"], "approved_by": action["approved_by"]}