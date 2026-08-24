"""
Roadmap item 6, Track 2: standalone demonstration endpoint for the
Parkinson's voice classifier trained in ml/training/train_parkinsons_classifier.py.

Deliberately decoupled from the live patient pipeline (per
claude/neuro-mas-architecture-v2.md's two-track ML design):
  - never reads or writes any patient/Supabase table
  - never called from app/orchestrator or any agent
  - requires login (so it's not a wide-open public endpoint) but the
    prediction is NOT tied to the caller's patient record in any way
  - always returns an explicit disclaimer in the response
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.schemas.ml_demo import ParkinsonsPredictionOut, ParkinsonsVoiceFeaturesIn
from app.services.auth import get_current_user

router = APIRouter(prefix="/demo", tags=["ml-demo"])

# app/api/routes/ml_demo.py -> parents[3] is the backend/ directory.
MODEL_DIR = Path(__file__).resolve().parents[3] / "ml" / "models"
MODEL_PATH = MODEL_DIR / "parkinsons_demo_classifier.joblib"
METADATA_PATH = MODEL_DIR / "parkinsons_demo_classifier_metadata.json"

_model = None
_metadata = None

DISCLAIMER = (
    "This is a DEMONSTRATION classifier trained on a public research dataset "
    "(UCI Parkinsons, 195 voice recordings from 31 people). It is NOT connected "
    "to any patient's real data, is NOT part of the NEURO-MAS clinical pipeline, "
    "and does NOT diagnose the person submitting this request. It exists solely "
    "to demonstrate a trained ML component."
)


def _load_model():
    global _model, _metadata
    if _model is None:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise HTTPException(
                503,
                "Demonstration classifier is not available -- it has not been trained yet. "
                "Run ml/training/train_parkinsons_classifier.py first.",
            )
        _model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    return _model, _metadata


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