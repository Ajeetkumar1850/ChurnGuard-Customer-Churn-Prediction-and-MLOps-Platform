"""Tests for the feature engineering pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import (
    ID_COLUMN,
    TARGET_COLUMN,
    build_feature_dataset,
    encode_target,
    split_feature_types,
    validate_feature_input,
)


def make_processed_df() -> pd.DataFrame:
    """Create a minimal processed churn-like DataFrame for testing."""
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
                "TotalCharges": 29.85,
                "Churn": "No",
            },
            {
                "customerID": "0002-B",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 85.50,
                "TotalCharges": 2052.00,
                "Churn": "Yes",
            },
        ]
    )


def test_validate_feature_input_passes_for_valid_dataframe() -> None:
    """Test that feature input validation passes for valid data."""
    df = make_processed_df()
    validate_feature_input(df)


def test_validate_feature_input_raises_for_missing_target() -> None:
    """Test that missing target column raises a ValueError."""
    df = make_processed_df().drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_feature_input(df)


def test_encode_target_maps_yes_no_to_binary() -> None:
    """Test that Yes/No churn values are encoded to 1/0."""
    df = make_processed_df()
    encoded_df = encode_target(df)

    assert set(encoded_df[TARGET_COLUMN].unique()) == {0, 1}


def test_split_feature_types_excludes_id_and_target() -> None:
    """Test that ID and target columns are excluded from feature type splits."""
    df = encode_target(make_processed_df())
    numeric_columns, categorical_columns = split_feature_types(df)

    assert ID_COLUMN not in numeric_columns
    assert ID_COLUMN not in categorical_columns
    assert TARGET_COLUMN not in numeric_columns
    assert TARGET_COLUMN not in categorical_columns
    assert "tenure" in numeric_columns
    assert "Contract" in categorical_columns


def test_build_feature_dataset_returns_binary_target_and_no_customer_id() -> None:
    """Test final feature dataset structure."""
    df = make_processed_df()
    feature_df = build_feature_dataset(df)

    assert TARGET_COLUMN in feature_df.columns
    assert ID_COLUMN not in feature_df.columns
    assert set(feature_df[TARGET_COLUMN].unique()) == {0, 1}


def test_build_feature_dataset_creates_encoded_columns() -> None:
    """Test that one-hot encoded columns are created for categoricals."""
    df = make_processed_df()
    feature_df = build_feature_dataset(df)

    encoded_columns = [column for column in feature_df.columns if column.startswith("Contract_")]
    assert len(encoded_columns) >= 1