"""FastAPI application for churn model inference."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from src.serving.predict import make_prediction

app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI service for telecom customer churn prediction.",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict:
    """Health-style root endpoint."""
    return {"message": "Customer Churn Prediction API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate a churn prediction from engineered feature inputs."""
    try:
        prediction, probability = make_prediction(request.features)
        prediction_label = "Yes" if prediction == 1 else "No"

        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            churn_probability=probability,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc