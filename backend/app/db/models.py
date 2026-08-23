"""
SQLAlchemy models -- the persisted half of the patient-state / audit spec
(sections 10, 21). Every autonomous workflow's inputs, agent outputs,
proposed actions and safety decisions land in these tables so a clinician
can reconstruct exactly what happened and why (spec section 21, "Complete
Audit Log").

Uses SQLite by default (zero-setup for an academic project) but only uses
portable SQLAlchemy types, so switching DATABASE_URL to Postgres is a
config change, not a rewrite (spec section 8: "PostgreSQL for structured
... data").
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    full_name: Mapped[str] = mapped_column(String, default="")
    role: Mapped[str] = mapped_column(String, default="patient")  # clinician | patient | caregiver
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(String, ForeignKey("users.id"), nullable=True)
    clinician_id: Mapped[str | None] = mapped_column(String, ForeignKey("users.id"), nullable=True)
    full_name: Mapped[str] = mapped_column(String)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[str | None] = mapped_column(String, nullable=True)
    preferred_language: Mapped[str | None] = mapped_column(String, nullable=True)
    notification_opt_in: Mapped[bool] = mapped_column(Boolean, default=True)
    clinician_restricted_actions: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    conditions: Mapped[list["Condition"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    observations: Mapped[list["Observation"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    medications: Mapped[list["Medication"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    allergies: Mapped[list["Allergy"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    metrics: Mapped[list["LongitudinalMetric"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class Condition(Base):
    __tablename__ = "conditions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"))
    disease_id: Mapped[str] = mapped_column(String)
    preferred_name: Mapped[str] = mapped_column(String)
    subtype: Mapped[str | None] = mapped_column(String, nullable=True)
    stage: Mapped[str | None] = mapped_column(String, nullable=True)
    diagnosed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="conditions")


class Observation(Base):
    __tablename__ = "observations"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"))
    workflow_id: Mapped[str | None] = mapped_column(String, nullable=True)
    modality: Mapped[str] = mapped_column(String)  # symptom | medication_log | wearable | vital | caregiver_report
    payload: Mapped[dict] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String, default="patient_reported")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_now)

    patient: Mapped["Patient"] = relationship(back_populates="observations")


class Medication(Base):
    __tablename__ = "medications"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"))
    name: Mapped[str] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    schedule: Mapped[str | None] = mapped_column(String, nullable=True)
    adherence_pct_30d: Mapped[float | None] = mapped_column(Float, nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="medications")


class Allergy(Base):
    __tablename__ = "allergies"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"))
    substance: Mapped[str] = mapped_column(String)

    patient: Mapped["Patient"] = relationship(back_populates="allergies")


class LongitudinalMetric(Base):
    __tablename__ = "longitudinal_metrics"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"))
    metric: Mapped[str] = mapped_column(String)
    baseline_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    window: Mapped[str] = mapped_column(String, default="unspecified")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    patient: Mapped["Patient"] = relationship(back_populates="metrics")


class Workflow(Base):
    __tablename__ = "workflows"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"))
    trigger_reason: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="running")
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict)


class AgentEventLog(Base):
    __tablename__ = "agent_event_log"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    workflow_id: Mapped[str] = mapped_column(String, ForeignKey("workflows.id"))
    patient_id: Mapped[str] = mapped_column(String)
    agent_name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"))
    workflow_id: Mapped[str | None] = mapped_column(String, nullable=True)
    severity: Mapped[str] = mapped_column(String)  # low | medium | high | critical
    reason: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String, default="open")  # open | acknowledged | escalated
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    acknowledged_by: Mapped[str | None] = mapped_column(String, nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ActionRecord(Base):
    __tablename__ = "actions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(String, ForeignKey("patients.id"))
    workflow_id: Mapped[str] = mapped_column(String)
    action_type: Mapped[str] = mapped_column(String)
    reason: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[list] = mapped_column(JSON, default=list)
    risk_level: Mapped[str] = mapped_column(String)
    autonomy_level: Mapped[int] = mapped_column(Integer)
    requires_human_approval: Mapped[bool] = mapped_column(Boolean)
    approved_by: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String)
    safety_reasons_json: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    workflow_id: Mapped[str] = mapped_column(String)
    patient_id: Mapped[str] = mapped_column(String)
    event: Mapped[str] = mapped_column(String)
    detail_json: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_now)
