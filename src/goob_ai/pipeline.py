"""Entry points for running the full Assignment 2 experiment."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from goob_ai import data, evaluation, modeling
from goob_ai.config import DataPaths, ExperimentConfig

logger = logging.getLogger(__name__)


def run_full_pipeline() -> dict[str, Any]:
    """Execute the canonical workflow used inside the notebook."""

    config = ExperimentConfig()
    paths = DataPaths()
    df = data.cached_feature_frame(paths, config)
    X, y = data.feature_target_split(df, config)

    cv_df = modeling.cross_validate_registry(X, y, config)
    holdout_results = modeling.holdout_evaluation(
        X,
        y,
        config,
        model_names=("log_reg", "random_forest", "hist_gb"),
    )
    best_result = max(holdout_results, key=lambda result: result.test_accuracy)
    logger.info("Best holdout model: %s", best_result.model_name)

    # Fit best estimator on entire dataset for downstream reporting.
    best_result.estimator.fit(X, y)
    artifacts = evaluation.capture_evaluation_artifacts(
        best_result.estimator,
        X,
        y,
        ["low", "medium", "high"],
    )

    return {
        "config": config,
        "paths": paths,
        "dataframe": df,
        "features": X,
        "target": y,
        "cv_results": cv_df,
        "holdout": holdout_results,
        "best_result": best_result,
        "artifacts": artifacts,
    }


def main() -> None:
    """Allow `uv run goob-ai` to execute the experiment quickly."""

    results = run_full_pipeline()
    cv_path = DataPaths().reports_dir / "cv_results.csv"
    holdout_path = DataPaths().reports_dir / "holdout_results.csv"
    results["cv_results"].to_csv(cv_path, index=False)
    pd.DataFrame([vars(item) for item in results["holdout"]]).drop(
        columns=["estimator"],
    ).to_csv(holdout_path, index=False)
    logger.info("Persisted cv summary to %s", cv_path)
    logger.info("Persisted holdout summary to %s", holdout_path)

