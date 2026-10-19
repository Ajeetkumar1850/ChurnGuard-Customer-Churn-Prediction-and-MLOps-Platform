"""Utility functions for loading YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed YAML contents as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML file is empty or invalid.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(f"Config file is empty or invalid: {path}")

    return config