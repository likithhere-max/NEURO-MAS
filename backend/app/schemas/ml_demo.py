"""
Roadmap item 6, Track 2: request/response schemas for the standalone
Parkinson's demonstration classifier endpoints. Deliberately separate from
app/schemas/api.py -- these schemas have nothing to do with real patients.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ParkinsonsVoiceFeaturesIn(BaseModel):
    """
    The 22 acoustic features from the UCI Parkinson's voice dataset, in the
    exact units/definitions used by that dataset. Field names use the
    dataset's own punctuation-heavy column names as JSON aliases (e.g.
    "MDVP:Fo(Hz)") so a caller sending a raw row from that dataset can use
    the field names as-is; the Python attribute names are just sanitized
    versions of the same thing.
    """
    model_config = {"populate_by_name": True, "allow_inf_nan": False}

    mdvp_fo_hz: float = Field(..., alias="MDVP:Fo(Hz)", description="Average vocal fundamental frequency (Hz)")
    mdvp_fhi_hz: float = Field(..., alias="MDVP:Fhi(Hz)", description="Maximum vocal fundamental frequency (Hz)")
    mdvp_flo_hz: float = Field(..., alias="MDVP:Flo(Hz)", description="Minimum vocal fundamental frequency (Hz)")
    mdvp_jitter_pct: float = Field(..., alias="MDVP:Jitter(%)")
    mdvp_jitter_abs: float = Field(..., alias="MDVP:Jitter(Abs)")
    mdvp_rap: float = Field(..., alias="MDVP:RAP")
    mdvp_ppq: float = Field(..., alias="MDVP:PPQ")
    jitter_ddp: float = Field(..., alias="Jitter:DDP")
    mdvp_shimmer: float = Field(..., alias="MDVP:Shimmer")
    mdvp_shimmer_db: float = Field(..., alias="MDVP:Shimmer(dB)")
    shimmer_apq3: float = Field(..., alias="Shimmer:APQ3")
    shimmer_apq5: float = Field(..., alias="Shimmer:APQ5")
    mdvp_apq: float = Field(..., alias="MDVP:APQ")
    shimmer_dda: float = Field(..., alias="Shimmer:DDA")
    nhr: float = Field(..., alias="NHR")
    hnr: float = Field(..., alias="HNR")
    rpde: float = Field(..., alias="RPDE")
    dfa: float = Field(..., alias="DFA")
    spread1: float = Field(..., alias="spread1")
    spread2: float = Field(..., alias="spread2")
    d2: float = Field(..., alias="D2")
    ppe: float = Field(..., alias="PPE")


class ParkinsonsPredictionOut(BaseModel):
    # When `abstained` is true the input was outside the validated domain: NO prediction is made and the
    # prediction fields are null. (Phase 2, C4: an unreliable number is never returned just to fill the schema.)
    predicted_class: int | None = None  # 0 = healthy, 1 = parkinsons (matches training label encoding)
    predicted_label: str | None = None
    probability_parkinsons: float | None = None
    probability_healthy: float | None = None
    abstained: bool = False
    domain: dict | None = None
    disclaimer: str


class ParkinsonsRiskFeaturesIn(BaseModel):
    """
    Inputs for the risk/worsening-trend classifier trained in
    ml/training/train_risk_classifier.py. Unlike the disease classifier
    above, these are CHANGE (delta) values relative to a patient's own
    baseline telemonitoring visit, not raw acoustic measurements -- that
    reframing is what made this model beat chance under proper patient-level
    validation (see the training script's docstring for why). Field names
    use the Telemonitoring dataset's own column names (no "MDVP:" prefix,
    unlike the voice dataset above) prefixed with "delta_", as JSON aliases.
    """
    model_config = {"populate_by_name": True, "allow_inf_nan": False}

    delta_jitter_pct: float = Field(..., alias="delta_Jitter(%)")
    delta_jitter_abs: float = Field(..., alias="delta_Jitter(Abs)")
    delta_jitter_rap: float = Field(..., alias="delta_Jitter:RAP")
    delta_jitter_ppq5: float = Field(..., alias="delta_Jitter:PPQ5")
    delta_jitter_ddp: float = Field(..., alias="delta_Jitter:DDP")
    delta_shimmer: float = Field(..., alias="delta_Shimmer")
    delta_shimmer_db: float = Field(..., alias="delta_Shimmer(dB)")
    delta_shimmer_apq3: float = Field(..., alias="delta_Shimmer:APQ3")
    delta_shimmer_apq5: float = Field(..., alias="delta_Shimmer:APQ5")
    delta_shimmer_apq11: float = Field(..., alias="delta_Shimmer:APQ11")
    delta_shimmer_dda: float = Field(..., alias="delta_Shimmer:DDA")
    delta_nhr: float = Field(..., alias="delta_NHR")
    delta_hnr: float = Field(..., alias="delta_HNR")
    delta_rpde: float = Field(..., alias="delta_RPDE")
    delta_dfa: float = Field(..., alias="delta_DFA")
    delta_ppe: float = Field(..., alias="delta_PPE")
    test_time: float = Field(..., description="Days since the patient's baseline telemonitoring visit")
    baseline_total_UPDRS: float = Field(..., description="The patient's total_UPDRS score at their baseline visit")


class ParkinsonsRiskPredictionOut(BaseModel):
    predicted_class: int | None = None  # 0 = stable/improving, 1 = worsening (matches training label encoding)
    predicted_label: str | None = None
    probability_worsening: float | None = None
    probability_stable_or_improving: float | None = None
    abstained: bool = False
    domain: dict | None = None
    disclaimer: str