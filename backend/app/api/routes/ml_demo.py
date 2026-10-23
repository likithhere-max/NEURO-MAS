"""
Roadmap item 6, Track 2: standalone demonstration endpoints for the
Parkinson's models trained in ml/training/train_parkinsons_classifier.py
(disease classifier) and ml/training/train_risk_classifier.py (risk/
worsening-trend classifier).

Deliberately decoupled from the live patient pipeline (per
claude/neuro-mas-architecture-v2.md's two-track ML design):
  - never reads or writes any patient/Supabase table
  - never called from app/orchestrator or any agent
  - requires login (so it's not a wide-open public endpoint) but the
    prediction is NOT tied to the caller's patient record in any way
  - always returns an explicit disclaimer in the response
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.schemas.ml_demo import (
    ParkinsonsPredictionOut,
    ParkinsonsRiskFeaturesIn,
    ParkinsonsRiskPredictionOut,
    ParkinsonsVoiceFeaturesIn,
)
from app.services.auth import get_current_user

router = APIRouter(prefix="/demo", tags=["ml-demo"])

# app/api/routes/ml_demo.py -> parents[3] is the backend/ directory.
MODEL_DIR = Path(__file__).resolve().parents[3] / "ml" / "models"

MODEL_PATH = MODEL_DIR / "parkinsons_demo_classifier.joblib"
METADATA_PATH = MODEL_DIR / "parkinsons_demo_classifier_metadata.json"

RISK_MODEL_PATH = MODEL_DIR / "parkinsons_risk_classifier.joblib"
RISK_METADATA_PATH = MODEL_DIR / "parkinsons_risk_classifier_metadata.json"

_model = None
_metadata = None
_risk_model = None
_risk_metadata = None

DISCLAIMER = (
    "This is a DEMONSTRATION classifier trained on a public research dataset "
    "(UCI Parkinsons, 195 voice recordings from 32 people). It is NOT connected "
    "to any patient's real data, is NOT part of the NEURO-MAS clinical pipeline, "
    "and does NOT diagnose the person submitting this request. It exists solely "
    "to demonstrate a trained ML component."
)

RISK_DISCLAIMER = (
    "This is a DEMONSTRATION classifier trained on a public research dataset "
    "(UCI Parkinson's Telemonitoring, 5,875 recordings from 42 patients). It "
    "predicts whether a patient's symptom trend (relative to their OWN baseline "
    "visit) is worsening, using a demonstration-only threshold that is NOT a "
    "validated clinical cutoff. It is NOT connected to any patient's real data, "
    "is NOT part of the NEURO-MAS clinical pipeline, and does NOT assess the "
    "person submitting this request. Its cross-patient accuracy (ROC-AUC ~0.64) "
    "is modest -- meaningfully better than chance, but far from clinical-grade. "
    "It exists solely to demonstrate a trained ML component evaluated honestly."
)


def _verified_load(path):
    """Load a joblib artifact only if its SHA-256 matches ml/models/MANIFEST.json (pickle = code execution)."""
    manifest_path = MODEL_DIR / "MANIFEST.json"
    expected = (json.loads(manifest_path.read_text()).get("sha256", {}) if manifest_path.exists() else {}).get(path.name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected or not hmac.compare_digest(expected, actual):
        raise HTTPException(503, "Model artifact failed integrity verification and was not loaded.")
    return joblib.load(path)


def _load_model():
    global _model, _metadata
    if _model is None:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise HTTPException(
                503,
                "Demonstration classifier is not available -- it has not been trained yet. "
                "Run ml/training/train_parkinsons_classifier.py first.",
            )
        _model = _verified_load(MODEL_PATH)
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    return _model, _metadata


def _load_risk_model():
    global _risk_model, _risk_metadata
    if _risk_model is None:
        if not RISK_MODEL_PATH.exists() or not RISK_METADATA_PATH.exists():
            raise HTTPException(
                503,
                "Demonstration risk classifier is not available -- it has not been trained yet. "
                "Run ml/training/train_risk_classifier.py first.",
            )
        _risk_model = _verified_load(RISK_MODEL_PATH)
        with open(RISK_METADATA_PATH) as f:
            _risk_metadata = json.load(f)
    return _risk_model, _risk_metadata


@router.post("/parkinsons-classifier", response_model=ParkinsonsPredictionOut)
def classify_parkinsons_voice_sample(payload: ParkinsonsVoiceFeaturesIn, user: dict = Depends(get_current_user)):
    model, metadata = _load_model()
    row = payload.model_dump(by_alias=True)
    X_row = pd.DataFrame([row])[metadata["feature_columns_in_order"]]

    predicted_class = int(model.predict(X_row)[0])
    probabilities = model.predict_proba(X_row)[0]  # [P(healthy), P(parkinsons)]

    return ParkinsonsPredictionOut(
        predicted_class=predicted_class,
        predicted_label="parkinsons" if predicted_class == 1 else "healthy",
        probability_healthy=round(float(probabilities[0]), 4),
        probability_parkinsons=round(float(probabilities[1]), 4),
        disclaimer=DISCLAIMER,
    )


@router.get("/parkinsons-classifier/info")
def parkinsons_classifier_info(user: dict = Depends(get_current_user)):
    """Returns the model's training metadata (dataset, metrics, thresholds) for transparency."""
    _, metadata = _load_model()
    return metadata


@router.post("/parkinsons-risk-classifier", response_model=ParkinsonsRiskPredictionOut)
def classify_parkinsons_risk_trend(payload: ParkinsonsRiskFeaturesIn, user: dict = Depends(get_current_user)):
    model, metadata = _load_risk_model()
    row = payload.model_dump(by_alias=True)
    X_row = pd.DataFrame([row])[metadata["feature_columns_in_order"]]

    predicted_class = int(model.predict(X_row)[0])
    probabilities = model.predict_proba(X_row)[0]  # [P(stable/improving), P(worsening)]

    return ParkinsonsRiskPredictionOut(
        predicted_class=predicted_class,
        predicted_label="worsening" if predicted_class == 1 else "stable_or_improving",
        probability_stable_or_improving=round(float(probabilities[0]), 4),
        probability_worsening=round(float(probabilities[1]), 4),
        disclaimer=RISK_DISCLAIMER,
    )


@router.get("/parkinsons-risk-classifier/info")
def parkinsons_risk_classifier_info(user: dict = Depends(get_current_user)):
    """Returns the model's training metadata (dataset, metrics, thresholds) for transparency."""
    _, metadata = _load_risk_model()
    return metadata