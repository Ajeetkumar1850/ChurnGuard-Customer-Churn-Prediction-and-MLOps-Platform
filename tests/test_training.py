"""Tests for the model training pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.evaluate import calculate_classification_metrics
from src.training.train import (
    TARGET_COLUMN,
    split_features_and_target,
    train_baseline_model,
)


def make_feature_df() -> pd.DataFrame:
    """Create a minimal feature-like dataset for testing."""
    return pd.DataFrame(
        [
            {"tenure": 1, "MonthlyCharges": 29.85, "TotalCharges": 29.85, "Contract_Month-to-month": 1, TARGET_COLUMN: 0},
            {"tenure": 24, "MonthlyCharges": 85.50, "TotalCharges": 2052.00, "Contract_Month-to-month": 0, TARGET_COLUMN: 1},
            {"tenure": 12, "MonthlyCharges": 56.10, "TotalCharges": 673.20, "Contract_Month-to-month": 1, TARGET_COLUMN: 0},
            {"tenure": 36, "MonthlyCharges": 99.40, "TotalCharges": 3578.40, "Contract_Month-to-month": 0, TARGET_COLUMN: 1},
        ]
    )


def test_split_features_and_target_returns_expected_shapes() -> None:
    """Test feature/target split behavior."""
    df = make_feature_df()

    X, y = split_features_and_target(df)

    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y) == 4


def test_calculate_classification_metrics_returns_expected_keys() -> None:
    """Test metric dictionary structure."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])

    metrics = calculate_classification_metrics(y_true, y_pred)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
    }

    assert expected_keys.issubset(metrics.keys())


def test_train_baseline_model_returns_fitted_model() -> None:
    """Test that baseline training returns a fitted pipeline."""
    df = make_feature_df()
    X, y = split_features_and_target(df)

    model = train_baseline_model(X, y)

    assert hasattr(model, "predict")
    assert "imputer" in model.named_steps
    assert "classifier" in model.named_steps
    assert hasattr(model.named_steps["classifier"], "coef_")