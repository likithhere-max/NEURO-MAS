from __future__ import annotations
from typing import Any, Optional
import math

from pydantic import BaseModel, EmailStr, Field, field_validator

# Bounds for the free-form observation dicts (vitals / wearable). They exist so that NaN/Infinity can never reach a
# threshold comparison (NaN > x is always False, which would read as "normal") and so a request cannot carry an
# arbitrarily large or deeply nested structure into the agents.
_MAX_KEYS, _MAX_KEY_LEN, _MAX_STR, _MAX_DEPTH, _MAX_LIST = 100, 100, 500, 3, 1000


def _check_value(v: Any, depth: int, path: str) -> None:
    if depth > _MAX_DEPTH:
        raise ValueError(f"{path}: nested deeper than {_MAX_DEPTH} levels")
    if v is None or isinstance(v, bool):
        return
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"{path}: non-finite number (NaN/Infinity) is not allowed")
        return
    if isinstance(v, str):
        if len(v) > _MAX_STR:
            raise ValueError(f"{path}: string longer than {_MAX_STR} characters")
        return
    if isinstance(v, dict):
        _check_dict(v, depth + 1, path)
        return
    if isinstance(v, (list, tuple)):
        if len(v) > _MAX_LIST:
            raise ValueError(f"{path}: list longer than {_MAX_LIST} items")
        for i, x in enumerate(v):
            _check_value(x, depth + 1, f"{path}[{i}]")
        return
    raise ValueError(f"{path}: unsupported value type {type(v).__name__}")


def _check_dict(d: dict, depth: int, path: str) -> None:
    if len(d) > _MAX_KEYS:
        raise ValueError(f"{path}: more than {_MAX_KEYS} keys")
    for k, v in d.items():
        if not isinstance(k, str) or len(k) > _MAX_KEY_LEN:
            raise ValueError(f"{path}: keys must be strings of at most {_MAX_KEY_LEN} characters")
        _check_value(v, depth, f"{path}.{k}")


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(default="", max_length=200)
    role: str = "patient"  # clinician | patient | caregiver
    invite_code: Optional[str] = Field(default=None, max_length=128)  # required for clinician signup in production
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
class PatientCreate(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    age: Optional[int] = Field(default=None, ge=0, le=130)
    sex: Optional[str] = Field(default=None, max_length=40)
    preferred_language: Optional[str] = Field(default=None, max_length=40)
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
    name: str = Field(max_length=300)
    severity: Optional[str] = Field(default=None, max_length=50)
    onset: Optional[str] = Field(default=None, max_length=100)
    free_text: Optional[str] = Field(default=None, max_length=5000)
class MedicationLogIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    taken: bool = True
    scheduled_time: Optional[str] = None
class ObservationBatchIn(BaseModel):
    symptoms: list[SymptomIn] = []
    medication_logs: list[MedicationLogIn] = []
    wearable: dict[str, Any] = {}
    vitals: dict[str, Any] = {}
    caregiver_notes: Optional[str] = Field(default=None, max_length=5000)

    @field_validator("wearable", "vitals")
    @classmethod
    def _bounded_finite(cls, v: dict, info) -> dict:
        _check_dict(v, 1, info.field_name)
        return v
class MedicationIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    active: bool = True
    schedule: Optional[str] = Field(default=None, max_length=200)
    # FRACTION in [0, 1] (0.85 == 85%), the unit every agent assumes. Percent values (e.g. 85) are rejected
    # rather than silently misread.
    adherence_pct_30d: Optional[float] = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
class WorkflowTriggerIn(BaseModel):
    trigger_reason: str = "manual_check_in"
    observations: ObservationBatchIn
class AlertAcknowledgeIn(BaseModel):
    note: Optional[str] = None
class KnowledgeSearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
class ConditionIn(BaseModel):
    disease_id: str
    preferred_name: str
    subtype: Optional[str] = None
    stage: Optional[str] = None
    diagnosed_at: Optional[str] = None  # ISO datetime string
class AllergyIn(BaseModel):
    substance: str = Field(min_length=1, max_length=200)
class LongitudinalMetricIn(BaseModel):
    metric: str = Field(min_length=1, max_length=100)
    baseline_value: Optional[float] = Field(default=None, allow_inf_nan=False)
    current_value: Optional[float] = Field(default=None, allow_inf_nan=False)
    window: str = Field(default="unspecified", max_length=100)
class MetricReadingIn(BaseModel):
    metric: str = Field(min_length=1, max_length=100)
    value: float = Field(allow_inf_nan=False)
    recorded_at: Optional[str] = None  # ISO datetime string; defaults to now() in DB if omitted