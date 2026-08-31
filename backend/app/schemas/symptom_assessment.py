"""
Natural-language symptom assessment schemas (new feature, 2-day review sprint).

Purpose: let a patient describe symptoms in their own words instead of
picking from a fixed disease list or hand-building a structured payload.
This module defines ONLY the extraction contract -- it does not change any
existing agent. The extraction result is converted (see
app/services/symptom_extraction.py) into the same `SymptomIn` list every
downstream agent (data_intake, neuro_diagnostic, risk_stratification,
knowledge, personalization) already consumes, so the multi-agent pipeline
itself is completely unmodified by this feature.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ExtractedSymptom(BaseModel):
    name: str
    location: Optional[str] = None
    onset: Optional[str] = None
    frequency: Optional[str] = None
    severity: Optional[str] = None  # "mild" | "moderate" | "severe" | None if not stated
    context: Optional[str] = None  # e.g. "primarily at rest", "worse in the evening"


class ExtractedSymptomProfile(BaseModel):
    """Gemini's structured output for free-text -> structured clinical features.

    Every field is either lifted directly from the patient's own words or
    left empty/None -- the extraction prompt (see symptom_extraction.py)
    explicitly forbids inventing a symptom, duration, or severity the
    patient did not state. This schema has no diagnosis/condition/risk
    field on purpose: extraction never gets to guess a disease.
    """

    symptoms: list[ExtractedSymptom] = Field(default_factory=list)
    associated_features: list[str] = Field(default_factory=list)
    duration: Optional[str] = None
    progression: Optional[str] = None  # "progressive" | "stable" | "improving" | "unknown"
    triggers: list[str] = Field(default_factory=list)
    relieving_factors: list[str] = Field(default_factory=list)
    relevant_history: list[str] = Field(default_factory=list)
    medications_mentioned: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(
        default_factory=list,
        description="What the extractor flagged as ambiguous or not stated, e.g. "
        "'exact onset date not given'. Never left implicit -- always explicit.",
    )


class SymptomAssessmentIn(BaseModel):
    free_text: str
    # Optional structured follow-up answers from the guided questions in the
    # UI (section 18 of the brief) -- merged into the extraction prompt as
    # additional, explicitly-labeled patient-provided context, never treated
    # as more authoritative than the free text.
    onset_answer: Optional[str] = None
    worsening_answer: Optional[str] = None
    severity_answer: Optional[str] = None
    frequency_answer: Optional[str] = None
    location_answer: Optional[str] = None
    triggers_answer: Optional[str] = None
    associated_symptoms_answer: Optional[str] = None


class SymptomAssessmentOut(BaseModel):
    id: str
    patient_id: str
    workflow_id: Optional[str] = None
    free_text: str
    extracted: ExtractedSymptomProfile
    created_at: str