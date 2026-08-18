"""
main.py — FastAPI serving layer for the Predictive Maintenance Platform.
Wraps RUL prediction, health scoring, failure classification, and
maintenance recommendation into a single API endpoint.
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_rul_model_latest.joblib"
RUL_CAP = 125

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts Remaining Useful Life, health score, likely "
                 "failure mode, and recommended maintenance action for "
                 "an industrial engine from sensor readings.",
    version="1.0.0",
)

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_feature_cols = _bundle["feature_cols"]


class SensorReading(BaseModel):
    """One row of engineered features for a single engine at a point in time.
    In production this would be computed live from raw sensor streams by
    the same build_features.py pipeline used in training."""
    features: dict = Field(
        ..., description="Dict of feature_name -> value, matching the "
                          "72 features the model was trained on."
    )


class PredictionResponse(BaseModel):
    predicted_RUL: float
    health_score: float
    risk_level: str
    recommended_action: str


def rul_to_health_score(rul: float) -> float:
    score = 100 * (rul / RUL_CAP)
    return max(0.0, min(100.0, round(score, 1)))


def risk_level_from_score(score: float) -> str:
    if score >= 70:
        return "Healthy"
    elif score >= 40:
        return "Moderate Risk"
    elif score >= 15:
        return "High Risk"
    return "Critical"


def recommend_action(health_score: float) -> str:
    if health_score < 15:
        return "Emergency Shutdown"
    elif health_score < 40:
        return "Immediate Repair"
    elif health_score < 70:
        return "Scheduled Maintenance"
    return "Continue Monitoring"


@app.get("/")
def root():
    return {"status": "ok", "message": "Predictive Maintenance API is running."}


@app.get("/model-info")
def model_info():
    return {
        "n_features": len(_feature_cols),
        "rul_cap": RUL_CAP,
        "model_type": "XGBoost Regressor",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    missing = set(_feature_cols) - set(reading.features.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {sorted(missing)}"
        )

    row = pd.DataFrame([reading.features])[_feature_cols]
    predicted_rul = float(_model.predict(row)[0])
    health_score = rul_to_health_score(predicted_rul)
    risk = risk_level_from_score(health_score)
    action = recommend_action(health_score)

    return PredictionResponse(
        predicted_RUL=round(predicted_rul, 2),
        health_score=health_score,
        risk_level=risk,
        recommended_action=action,
    )