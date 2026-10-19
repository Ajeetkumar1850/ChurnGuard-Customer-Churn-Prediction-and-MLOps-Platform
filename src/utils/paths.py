"""Path utilities for project directories and files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.utils.config import load_yaml_config


def get_paths(config_path: str = "configs/paths.yaml") -> Dict[str, Path]:
    """Load project paths from YAML and return them as Path objects.

    Args:
        config_path: Path to the YAML path configuration file.

    Returns:
        Dictionary of resolved Path objects.
    """
    config = load_yaml_config(config_path)

    paths = {
        "raw_dir": Path(config["data"]["raw_dir"]),
        "interim_dir": Path(config["data"]["interim_dir"]),
        "processed_dir": Path(config["data"]["processed_dir"]),
        "raw_churn_data": Path(config["files"]["raw_churn_data"]),
        "processed_churn_data": Path(config["files"]["processed_churn_data"]),
        "feature_data": Path(config["files"]["feature_data"]),
        "ingestion_log": Path(config["files"]["ingestion_log"]),
        "feature_log": Path(config["files"]["feature_log"]),
        "training_metrics": Path(config["files"]["training_metrics"]),
        "baseline_model": Path(config["files"]["baseline_model"]),
        "models_dir": Path(config["artifacts"]["models_dir"]),
        "reports_dir": Path(config["artifacts"]["reports_dir"]),
        "figures_dir": Path(config["artifacts"]["figures_dir"]),
    }

    return paths


def ensure_directories_exist(paths: Dict[str, Path]) -> None:
    """Ensure required directories exist.

    Args:
        paths: Dictionary of project paths.
    """
    directory_keys = [
        "raw_dir",
        "interim_dir",
        "processed_dir",
        "models_dir",
        "reports_dir",
        "figures_dir",
    ]

    for key in directory_keys:
        paths[key].mkdir(parents=True, exist_ok=True)

    paths["ingestion_log"].parent.mkdir(parents=True, exist_ok=True)
    paths["feature_log"].parent.mkdir(parents=True, exist_ok=True)
    paths["training_metrics"].parent.mkdir(parents=True, exist_ok=True)
    paths["baseline_model"].parent.mkdir(parents=True, exist_ok=True)