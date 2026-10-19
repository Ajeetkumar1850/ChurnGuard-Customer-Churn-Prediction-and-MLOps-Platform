"""Prediction utilities for the churn model serving layer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

MODEL_PATH = Path("artifacts/models/baseline_model.joblib")


def load_model(model_path: str | Path = MODEL_PATH):
    """Load a trained model artifact.

    Args:
        model_path: Path to the saved model artifact.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def make_prediction(input_data: Dict[str, float]) -> Tuple[int, float]:
    """Generate a churn prediction from a single feature row.

    Args:
        input_data: Dictionary of model feature values.

    Returns:
        Tuple of:
            - predicted class (0 or 1)
            - positive-class probability
    """
    model = load_model()

    input_df = pd.DataFrame([input_data])
    predicted_class = int(model.predict(input_df)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
    else:
        probability = 0.0

    return predicted_class, probability