"""Model training and evaluation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from goob_ai.config import ExperimentConfig

logger = logging.getLogger(__name__)


def _numeric_pipeline() -> ColumnTransformer:
    """Return a transformation pipeline for numeric-only datasets."""

    scaler = StandardScaler()
    transformer = ColumnTransformer(
        transformers=[("num", scaler, slice(None))],
        remainder="drop",
    )
    return transformer


def build_model_registry(config: ExperimentConfig) -> Dict[str, BaseEstimator]:
    """Instantiate all estimators used for experimentation."""

    numeric_transformer = _numeric_pipeline()

    def with_pipeline(estimator: BaseEstimator) -> Pipeline:
        return Pipeline(
            steps=[
                ("numeric", numeric_transformer),
                ("model", estimator),
            ],
        )

    registry: Dict[str, BaseEstimator] = {
        "dummy_majority": with_pipeline(
            DummyClassifier(strategy="most_frequent"),
        ),
        "log_reg": with_pipeline(
            LogisticRegression(
                max_iter=500,
                random_state=config.random_state,
                solver="lbfgs",
            ),
        ),
        "random_forest": with_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                random_state=config.random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        "hist_gb": with_pipeline(
            HistGradientBoostingClassifier(
                random_state=config.random_state,
                max_depth=6,
                learning_rate=0.08,
            ),
        ),
    }
    return registry


def summarize_cv_results(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Convert cross_validate outputs into mean ± std summaries."""

    summary: dict[str, float] = {}
    for metric, values in scores.items():
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
    return summary


def cross_validate_registry(
    X: pd.DataFrame,
    y: pd.Series,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Cross-validate every model and return a tidy dataframe."""

    registry = build_model_registry(config)
    cv = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    rows: list[dict[str, float]] = []
    for model_name, estimator in registry.items():
        logger.info("Cross-validating %s", model_name)
        scores = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=config.scoring_metrics,
            n_jobs=-1,
            return_train_score=False,
        )
        summary = summarize_cv_results(scores)
        summary["model"] = model_name
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(by="test_accuracy_mean", ascending=False)


@dataclass
class HoldoutResult:
    """Captures evaluation metrics for a single holdout split."""

    model_name: str
    test_accuracy: float
    test_f1_macro: float
    test_balanced_accuracy: float
    estimator: BaseEstimator


def holdout_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    config: ExperimentConfig,
    *,
    model_names: Iterable[str] | None = None,
) -> list[HoldoutResult]:
    """Evaluate selected models on a single holdout split."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    registry = build_model_registry(config)
    if model_names is not None:
        registry = {name: registry[name] for name in model_names}

    results: list[HoldoutResult] = []
    for model_name, estimator in registry.items():
        logger.info("Fitting %s on holdout split", model_name)
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        result = HoldoutResult(
            model_name=model_name,
            test_accuracy=accuracy_score(y_test, predictions),
            test_f1_macro=f1_score(y_test, predictions, average="macro"),
            test_balanced_accuracy=balanced_accuracy_score(y_test, predictions),
            estimator=estimator,
        )
        results.append(result)
    return results

