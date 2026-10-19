"""MLflow utilities for experiment tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import mlflow


def set_local_mlflow_tracking() -> str:
    """Set MLflow tracking to a local file-based store.

    Returns:
        The tracking URI string.
    """
    tracking_dir = Path("mlruns").resolve()
    tracking_uri = tracking_dir.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def set_experiment(experiment_name: str) -> None:
    """Set the active MLflow experiment.

    Args:
        experiment_name: Name of the experiment.
    """
    mlflow.set_experiment(experiment_name)


def log_params(params: Dict[str, Any]) -> None:
    """Log parameter dictionary to MLflow.

    Args:
        params: Parameters to log.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: Dict[str, float]) -> None:
    """Log numeric metrics to MLflow.

    Args:
        metrics: Metrics to log.
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, float(value))