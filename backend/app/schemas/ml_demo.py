"""
Roadmap item 6, Track 2: request/response schemas for the standalone
Parkinson's demonstration classifier endpoint. Deliberately separate from
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
    model_config = {"populate_by_name": True}

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
    predicted_class: int  # 0 = healthy, 1 = parkinsons (matches training label encoding)
    predicted_label: str
    probability_parkinsons: float
    probability_healthy: float
    disclaimer: str