"""Pydantic schemas for the churn prediction API."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Schema for a single churn prediction request.

    This API version expects already-engineered feature inputs.
    """

    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of engineered feature names and numeric values.",
        examples=[
            {
                "tenure": 12.0,
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20,
                "SeniorCitizen": 0.0,
                "gender_Female": 1.0,
                "gender_Male": 0.0,
                "Partner_No": 0.0,
                "Partner_Yes": 1.0,
                "Dependents_No": 1.0,
                "Dependents_Yes": 0.0,
                "PhoneService_No": 0.0,
                "PhoneService_Yes": 1.0,
                "PaperlessBilling_No": 0.0,
                "PaperlessBilling_Yes": 1.0,
                "Contract_Month-to-month": 1.0,
                "Contract_One year": 0.0,
                "Contract_Two year": 0.0,
            }
        ],
    )


class PredictionResponse(BaseModel):
    """Schema for churn prediction output."""

    prediction: int
    prediction_label: str
    churn_probability: float