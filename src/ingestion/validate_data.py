"""Validation and cleaning functions for churn data ingestion."""

from __future__ import annotations

from typing import List

import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that all required churn columns exist.

    Args:
        df: Input DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the churn dataset for downstream use.

    Cleaning steps:
        - Copy the input data
        - Strip whitespace from column names
        - Convert TotalCharges to numeric
        - Drop rows with missing target
        - Drop duplicate rows
        - Reset index

    Args:
        df: Raw churn DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    cleaned_df = df.copy()

    cleaned_df.columns = cleaned_df.columns.str.strip()

    # IBM churn data often stores TotalCharges as object due to blank strings.
    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"], errors="coerce"
    )

    cleaned_df["customerID"] = cleaned_df["customerID"].astype(str).str.strip()
    cleaned_df["Churn"] = cleaned_df["Churn"].astype(str).str.strip()

    cleaned_df = cleaned_df.dropna(subset=["Churn"])
    cleaned_df = cleaned_df.drop_duplicates()
    cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df