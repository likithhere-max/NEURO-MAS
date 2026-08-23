from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models as m
from app.db.session import get_db
from app.schemas.api import AlertAcknowledgeIn
from app.services.auth import get_current_user

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, payload: AlertAcknowledgeIn, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    alert = db.get(m.Alert, alert_id)
    if not alert:
        raise HTTPException(404, "alert not found")
    alert.status = "acknowledged"
    alert.acknowledged_by = user.id
    alert.acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    return {"id": alert.id, "status": alert.status}


@router.get("")
def list_alerts(status_filter: str | None = None, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    q = db.query(m.Alert)
    if status_filter:
        q = q.filter(m.Alert.status == status_filter)
    alerts = q.order_by(m.Alert.created_at.desc()).all()
    return [{
        "id": a.id, "patient_id": a.patient_id, "severity": a.severity, "reason": a.reason,
        "status": a.status, "created_at": a.created_at.isoformat(),
    } for a in alerts]


@router.post("/actions/{action_id}/approve")
def approve_action(action_id: str, db: Session = Depends(get_db), user: m.User = Depends(get_current_user)):
    if user.role != "clinician":
        raise HTTPException(403, "only a clinician may authorize a restricted action")
    action = db.get(m.ActionRecord, action_id)
    if not action:
        raise HTTPException(404, "action not found")
    action.status = "executed"
    action.approved_by = user.id
    db.commit()
    return {"id": action.id, "status": action.status, "approved_by": action.approved_by}
