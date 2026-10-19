"""Data loading functions for the churn ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_churn_data(file_path: str | Path) -> pd.DataFrame:
    """Load the raw churn CSV file.

    Args:
        file_path: Path to the raw churn CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If the CSV is empty.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Raw data file is empty: {path}")

    return df