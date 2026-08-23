from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str = ""
    role: str = "patient"  # clinician | patient | caregiver


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class PatientCreate(BaseModel):
    full_name: str
    age: Optional[int] = None
    sex: Optional[str] = None
    preferred_language: Optional[str] = None
    conditions: list[str] = []  # disease_id or preferred_name from the knowledge base
    clinician_id: Optional[str] = None


class PatientOut(BaseModel):
    id: str
    full_name: str
    age: Optional[int]
    sex: Optional[str]
    conditions: list[str]

    model_config = {"from_attributes": True}


class SymptomIn(BaseModel):
    name: str
    severity: Optional[str] = None
    onset: Optional[str] = None
    free_text: Optional[str] = None


class MedicationLogIn(BaseModel):
    name: str
    taken: bool = True
    scheduled_time: Optional[str] = None


class ObservationBatchIn(BaseModel):
    symptoms: list[SymptomIn] = []
    medication_logs: list[MedicationLogIn] = []
    wearable: dict[str, Any] = {}
    vitals: dict[str, Any] = {}
    caregiver_notes: Optional[str] = None


class MedicationIn(BaseModel):
    name: str
    active: bool = True
    schedule: Optional[str] = None
    adherence_pct_30d: Optional[float] = None


class WorkflowTriggerIn(BaseModel):
    trigger_reason: str = "manual_check_in"
    observations: ObservationBatchIn


class AlertAcknowledgeIn(BaseModel):
    note: Optional[str] = None


class KnowledgeSearchIn(BaseModel):
    query: str
    top_k: int = 5
