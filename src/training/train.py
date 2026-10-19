"""Model training pipeline for the customer churn platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.training.evaluate import calculate_classification_metrics
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories_exist, get_paths

logger = get_logger(__name__)

TARGET_COLUMN = "Churn"


def load_feature_data(file_path: str | Path) -> pd.DataFrame:
    """Load the feature-engineered churn dataset from parquet.

    Args:
        file_path: Path to the feature parquet dataset.

    Returns:
        Loaded feature DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the dataset is empty.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Feature data file not found: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"Feature data file is empty: {path}")

    return df


def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a feature dataset into X and y.

    Args:
        df: Feature DataFrame containing the target column.

    Returns:
        Tuple of (X, y).

    Raises:
        ValueError: If the target column is missing.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    return X, y


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a Logistic Regression baseline model with imputation.

    Args:
        X_train: Training features.
        y_train: Training target values.

    Returns:
        Trained scikit-learn Pipeline.
    """
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    """Save training metrics to JSON.

    Args:
        metrics: Metrics dictionary.
        output_path: Output JSON path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_model(model: Pipeline, output_path: str | Path) -> None:
    """Save a trained model to disk.

    Args:
        model: Trained model instance.
        output_path: Output model path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def run_training_pipeline() -> None:
    """Run the end-to-end baseline training pipeline."""
    paths = get_paths()
    ensure_directories_exist(paths)

    training_config = load_yaml_config("configs/training.yaml")
    random_state = training_config["random_state"]
    test_size = training_config["split"]["test_size"]

    logger.info("Starting model training pipeline.")
    logger.info("Loading feature dataset from %s", paths["feature_data"])

    feature_df = load_feature_data(paths["feature_data"])
    logger.info("Feature dataset loaded with shape %s", feature_df.shape)

    X, y = split_features_and_target(feature_df)

    missing_value_count = int(X.isna().sum().sum())
    logger.info("Total missing feature values detected: %s", missing_value_count)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Training set shape: %s", X_train.shape)
    logger.info("Test set shape: %s", X_test.shape)

    model = train_baseline_model(X_train, y_train)
    logger.info("Baseline Logistic Regression pipeline trained successfully.")

    y_pred = model.predict(X_test)
    metrics = calculate_classification_metrics(y_test, y_pred)
    metrics["model_name"] = "logistic_regression_baseline"
    metrics["train_rows"] = int(X_train.shape[0])
    metrics["test_rows"] = int(X_test.shape[0])
    metrics["feature_count"] = int(X_train.shape[1])
    metrics["missing_feature_values"] = missing_value_count

    save_metrics(metrics, paths["training_metrics"])
    save_model(model, paths["baseline_model"])

    logger.info("Training metrics written to %s", paths["training_metrics"])
    logger.info("Baseline model written to %s", paths["baseline_model"])
    logger.info("Model training pipeline completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()