"""Hyperparameter tuning and experiment tracking for churn models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.training.evaluate import calculate_classification_metrics
from src.training.train import TARGET_COLUMN, load_feature_data, split_features_and_target
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories_exist, get_paths
from src.registry.mlflow_registry import (
    log_metrics,
    log_params,
    set_experiment,
    set_local_mlflow_tracking,
)

logger = get_logger(__name__)


def build_xgboost_pipeline(config: Dict[str, Any]) -> Pipeline:
    """Build an XGBoost training pipeline with imputation.

    Args:
        config: Training configuration dictionary.

    Returns:
        scikit-learn Pipeline containing imputer + XGBoost classifier.
    """
    xgb_config = config["xgboost"]

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                XGBClassifier(
                    objective=xgb_config["objective"],
                    eval_metric=xgb_config["eval_metric"],
                    n_estimators=xgb_config["n_estimators"],
                    max_depth=xgb_config["max_depth"],
                    learning_rate=xgb_config["learning_rate"],
                    subsample=xgb_config["subsample"],
                    colsample_bytree=xgb_config["colsample_bytree"],
                    random_state=config["random_state"],
                ),
            ),
        ]
    )
    return pipeline


def build_param_grid(config: Dict[str, Any]) -> Dict[str, list]:
    """Build GridSearchCV parameter grid for the XGBoost pipeline.

    Args:
        config: Training configuration dictionary.

    Returns:
        Parameter grid using pipeline-compatible parameter names.
    """
    grid = config["tuning"]["xgboost_param_grid"]

    return {
        "classifier__n_estimators": grid["n_estimators"],
        "classifier__max_depth": grid["max_depth"],
        "classifier__learning_rate": grid["learning_rate"],
    }


def save_tuning_summary(summary: Dict[str, Any], output_path: str | Path) -> None:
    """Save tuning summary to JSON.

    Args:
        summary: Summary dictionary.
        output_path: Output file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def run_tuning_pipeline() -> None:
    """Run XGBoost tuning and MLflow tracking."""
    paths = get_paths()
    ensure_directories_exist(paths)

    config = load_yaml_config("configs/training.yaml")
    random_state = config["random_state"]
    test_size = config["split"]["test_size"]
    cv = config["tuning"]["cv"]
    scoring = config["tuning"]["scoring"]

    logger.info("Starting hyperparameter tuning pipeline.")
    logger.info("Loading feature dataset from %s", paths["feature_data"])

    feature_df = load_feature_data(paths["feature_data"])
    X, y = split_features_and_target(feature_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_xgboost_pipeline(config)
    param_grid = build_param_grid(config)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    tracking_uri = set_local_mlflow_tracking()
    set_experiment("customer-churn-platform")

    logger.info("MLflow tracking URI set to %s", tracking_uri)
    logger.info("Beginning GridSearchCV tuning for XGBoost.")

    with mlflow.start_run(run_name="xgboost_gridsearch") as run:
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = calculate_classification_metrics(y_test, y_pred)

        log_params(
            {
                "model_type": "xgboost",
                "cv": cv,
                "scoring": scoring,
                **grid_search.best_params_,
            }
        )
        log_metrics(metrics)
        mlflow.log_metric("best_cv_score", float(grid_search.best_score_))

        mlflow.sklearn.log_model(best_model, artifact_path="model")

        summary = {
            "model_name": "xgboost_tuned",
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "test_metrics": metrics,
            "mlflow_run_id": run.info.run_id,
        }

    tuning_report_path = paths["reports_dir"] / "tuning_summary.json"
    tuned_model_path = paths["models_dir"] / "xgboost_tuned_model.joblib"

    save_tuning_summary(summary, tuning_report_path)
    joblib.dump(grid_search.best_estimator_, tuned_model_path)

    logger.info("Tuning summary written to %s", tuning_report_path)
    logger.info("Tuned XGBoost model written to %s", tuned_model_path)
    logger.info("Hyperparameter tuning pipeline completed successfully.")


if __name__ == "__main__":
    run_tuning_pipeline()