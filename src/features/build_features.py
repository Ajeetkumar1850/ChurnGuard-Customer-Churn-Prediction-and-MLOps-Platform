"""Feature engineering pipeline for the customer churn platform."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import ensure_directories_exist, get_paths

logger = get_logger(__name__)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

#loading data
def load_processed_data(file_path: str | Path) -> pd.DataFrame:
    """Load the processed churn dataset from parquet.

    Args:
        file_path: Path to the processed parquet dataset.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
        ValueError: If the dataset is empty.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"Processed data file is empty: {path}")

    return df


def validate_feature_input(df: pd.DataFrame) -> None:
    """Validate that the processed dataset contains required columns.

    Args:
        df: Input processed DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns for feature engineering: "
            f"{sorted(missing_columns)}"
        )


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the churn target from Yes/No to 1/0.

    Args:
        df: Processed DataFrame containing the Churn column.

    Returns:
        DataFrame with binary target encoding.

    Raises:
        ValueError: If unexpected target values are found.
    """
    encoded_df = df.copy()

    churn_mapping = {"Yes": 1, "No": 0}
    unique_values = set(encoded_df[TARGET_COLUMN].dropna().unique())

    if not unique_values.issubset(churn_mapping.keys()):
        raise ValueError(
            f"Unexpected target values found in {TARGET_COLUMN}: "
            f"{sorted(unique_values)}"
        )

    encoded_df[TARGET_COLUMN] = encoded_df[TARGET_COLUMN].map(churn_mapping)

    return encoded_df


def split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical feature columns.

    Excludes:
        - customerID
        - Churn target

    Args:
        df: Input DataFrame.

    Returns:
        Tuple containing:
            - numeric feature column names
            - categorical feature column names
    """
    feature_df = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])

    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    return numeric_columns, categorical_columns


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature dataset.

    Steps:
        - encode target
        - separate numeric and categorical columns
        - one-hot encode categorical columns
        - preserve numeric columns
        - append binary target column

    Args:
        df: Processed churn DataFrame.

    Returns:
        Model-ready feature DataFrame.
    """
    encoded_df = encode_target(df)
    numeric_columns, categorical_columns = split_feature_types(encoded_df)

    numeric_df = encoded_df[numeric_columns].copy()

    categorical_df = pd.get_dummies(
        encoded_df[categorical_columns],
        drop_first=False,
        dtype=int,
    )

    feature_df = pd.concat([numeric_df, categorical_df], axis=1)
    feature_df[TARGET_COLUMN] = encoded_df[TARGET_COLUMN].astype(int)

    return feature_df


def save_feature_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save the feature-engineered dataset to parquet.

    Args:
        df: Feature-engineered DataFrame.
        output_path: Path to the output parquet file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_feature_summary(
    input_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write a JSON summary of feature engineering results.

    Args:
        input_df: Original processed input DataFrame.
        feature_df: Final feature-engineered DataFrame.
        output_path: Path to the output JSON file.
    """
    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(input_df.shape[0]),
        "input_columns": int(input_df.shape[1]),
        "feature_rows": int(feature_df.shape[0]),
        "feature_columns": int(feature_df.shape[1]),
        "target_column": TARGET_COLUMN,
        "output_file": str(output_path).replace("feature_summary.json", "churn_features.parquet"),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def run_feature_pipeline() -> None:
    """Run the end-to-end feature engineering pipeline."""
    paths = get_paths()
    ensure_directories_exist(paths)

    logger.info("Starting feature engineering pipeline.")
    logger.info("Loading processed dataset from %s", paths["processed_churn_data"])

    processed_df = load_processed_data(paths["processed_churn_data"])
    validate_feature_input(processed_df)

    logger.info("Processed dataset loaded with shape %s", processed_df.shape)

    feature_df = build_feature_dataset(processed_df)
    logger.info("Feature dataset built with shape %s", feature_df.shape)

    save_feature_dataset(feature_df, paths["feature_data"])
    logger.info("Feature dataset written to %s", paths["feature_data"])

    save_feature_summary(processed_df, feature_df, paths["feature_log"])
    logger.info("Feature summary written to %s", paths["feature_log"])
    logger.info("Feature engineering pipeline completed successfully.")


if __name__ == "__main__":
    run_feature_pipeline()