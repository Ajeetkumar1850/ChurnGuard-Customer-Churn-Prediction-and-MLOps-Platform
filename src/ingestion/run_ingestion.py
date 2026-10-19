"""Entrypoint for the churn data ingestion pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.ingestion.load_data import load_raw_churn_data
from src.ingestion.validate_data import clean_churn_data, validate_required_columns
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories_exist, get_paths

logger = get_logger(__name__)


def run_ingestion() -> None:
    """Run the end-to-end ingestion pipeline."""
    paths = get_paths()
    ensure_directories_exist(paths)

    logger.info("Starting ingestion pipeline.")
    logger.info("Loading raw churn dataset from %s", paths["raw_churn_data"])

    raw_df = load_raw_churn_data(paths["raw_churn_data"])
    validate_required_columns(raw_df)

    logger.info("Raw dataset loaded with shape %s", raw_df.shape)

    cleaned_df = clean_churn_data(raw_df)
    logger.info("Cleaned dataset shape: %s", cleaned_df.shape)

    cleaned_df.to_parquet(paths["processed_churn_data"], index=False)
    logger.info("Processed dataset written to %s", paths["processed_churn_data"])

    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": int(raw_df.shape[0]),
        "raw_columns": int(raw_df.shape[1]),
        "processed_rows": int(cleaned_df.shape[0]),
        "processed_columns": int(cleaned_df.shape[1]),
        "output_file": str(paths["processed_churn_data"]),
    }

    with paths["ingestion_log"].open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    logger.info("Ingestion summary written to %s", paths["ingestion_log"])
    logger.info("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    run_ingestion()