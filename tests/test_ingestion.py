"""Tests for the data ingestion pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.validate_data import (
    REQUIRED_COLUMNS,
    clean_churn_data,
    validate_required_columns,
)


def make_sample_df() -> pd.DataFrame:
    """Create a minimal valid churn-like DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "customerID": "0001-A",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
                "Churn": "No",
            }
        ]
    )


def test_validate_required_columns_passes_for_valid_dataframe() -> None:
    """Test that required column validation passes for a valid DataFrame."""
    df = make_sample_df()
    validate_required_columns(df)


def test_validate_required_columns_raises_for_missing_column() -> None:
    """Test that missing required columns raise a ValueError."""
    df = make_sample_df().drop(columns=["Contract"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df)


def test_clean_churn_data_converts_totalcharges_to_numeric() -> None:
    """Test that TotalCharges is converted to numeric."""
    df = make_sample_df()
    cleaned_df = clean_churn_data(df)

    assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])


def test_required_columns_constant_is_complete() -> None:
    """Test that the project expects the IBM churn schema columns."""
    assert "customerID" in REQUIRED_COLUMNS
    assert "TotalCharges" in REQUIRED_COLUMNS
    assert "Churn" in REQUIRED_COLUMNS
    assert len(REQUIRED_COLUMNS) == 21