"""
Formal, persisted patient-state model.

This is the shared "blackboard" every agent reads from and writes findings
about. It intentionally treats every field as optional/partial: the whole
point of section 4 of the spec is that the system must degrade gracefully
when data is missing, not assume a fully populated record.

The personalization state vector P_t = [D_t, S_t, M_t, R_t, B_t, C_t, A_t, E_t]
(section 36) is represented directly as named sub-objects on PatientState
rather than a bare vector, so every dimension is inspectable and auditable
instead of being an opaque array of floats.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class Demographics(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None


class DiseaseState(BaseModel):
    """D_t"""
    conditions: list[str] = Field(default_factory=list)
    subtypes: dict[str, str] = Field(default_factory=dict)
    stage: dict[str, str] = Field(default_factory=dict)


class SymptomState(BaseModel):
    """S_t"""
    active_symptoms: list[dict[str, Any]] = Field(default_factory=list)
    severity_scores: dict[str, float] = Field(default_factory=dict)


class MedicationState(BaseModel):
    """M_t"""
    active_medications: list[dict[str, Any]] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    adherence_pct_30d: Optional[float] = None


class RiskState(BaseModel):
    """R_t"""
    current_risk: dict[str, Any] = Field(default_factory=dict)


class BehavioralState(BaseModel):
    """B_t - lifestyle / behavior"""
    sleep_hours_avg_7d: Optional[float] = None
    activity_level: Optional[str] = None
    mood: Optional[str] = None


class CognitiveState(BaseModel):
    """C_t"""
    cognitive_scores: dict[str, float] = Field(default_factory=dict)
    mobility_notes: Optional[str] = None
    tremor_notes: Optional[str] = None


class AdherenceState(BaseModel):
    """A_t"""
    monitoring_adherence_pct: Optional[float] = None
    missed_checkins_7d: int = 0


class EnvironmentalState(BaseModel):
    """E_t - environmental / contextual"""
    caregiver_present: Optional[bool] = None
    context_notes: Optional[str] = None


class LongitudinalTrend(BaseModel):
    metric: str
    baseline_value: Optional[float] = None
    current_value: Optional[float] = None
    trend: str = "unknown"  # "improving" | "stable" | "deteriorating" | "unknown"
    delta: Optional[float] = None
    window: str = "unspecified"


class PatientPreferences(BaseModel):
    preferred_language: Optional[str] = None
    communication_channel: Optional[str] = None
    notification_opt_in: bool = True


class ClinicianConstraints(BaseModel):
    restricted_actions: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class PatientState(BaseModel):
    """The full longitudinal, personalized patient state ("the blackboard")."""

    patient_id: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    demographics: Demographics = Field(default_factory=Demographics)
    disease_state: DiseaseState = Field(default_factory=DiseaseState)
    symptom_state: SymptomState = Field(default_factory=SymptomState)
    medication_state: MedicationState = Field(default_factory=MedicationState)
    risk_state: RiskState = Field(default_factory=RiskState)
    behavioral_state: BehavioralState = Field(default_factory=BehavioralState)
    cognitive_state: CognitiveState = Field(default_factory=CognitiveState)
    adherence_state: AdherenceState = Field(default_factory=AdherenceState)
    environmental_state: EnvironmentalState = Field(default_factory=EnvironmentalState)

    comorbidities: list[str] = Field(default_factory=list)
    seizure_history: list[dict[str, Any]] = Field(default_factory=list)
    previous_neurological_events: list[dict[str, Any]] = Field(default_factory=list)

    longitudinal_trends: list[LongitudinalTrend] = Field(default_factory=list)
    wearable_metrics: dict[str, Any] = Field(default_factory=dict)

    patient_preferences: PatientPreferences = Field(default_factory=PatientPreferences)
    clinician_constraints: ClinicianConstraints = Field(default_factory=ClinicianConstraints)

    current_alerts: list[str] = Field(default_factory=list)
    active_workflows: list[str] = Field(default_factory=list)

    data_completeness: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of expected state fields with real (non-null) data. "
                    "Used by agents/consensus to discount low-quality inputs.",
    )

    def compute_completeness(self) -> float:
        """Best-effort data-quality signal used to weight agent confidence.

        This is a simple heuristic (fraction of populated top-level
        sub-states), NOT a validated clinical data-quality metric.
        """
        checks = [
            bool(self.demographics.age),
            bool(self.disease_state.conditions),
            bool(self.symptom_state.active_symptoms),
            bool(self.medication_state.active_medications),
            self.medication_state.adherence_pct_30d is not None,
            bool(self.longitudinal_trends),
            bool(self.wearable_metrics),
            self.cognitive_state.cognitive_scores != {},
        ]
        return round(sum(1 for c in checks if c) / len(checks), 2)
