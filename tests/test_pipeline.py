"""End-to-end tests for the goob_ai pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from goob_ai import data, modeling
from goob_ai.config import DataPaths, ExperimentConfig

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def test_cached_feature_frame_contains_labels() -> None:
    """Ensure the cached feature frame includes the engineered label column."""

    config = ExperimentConfig()
    paths = DataPaths()
    df = data.cached_feature_frame(paths, config)
    assert not df.empty
    assert config.score_column in df.columns
    assert config.target_column in df.columns
    assert set(df[config.target_column].unique()) == {"low", "medium", "high"}


def test_model_registry_cross_validation_returns_metrics() -> None:
    """Cross-validation should produce results for every configured metric."""

    config = ExperimentConfig(n_splits=3)
    paths = DataPaths()
    df = data.cached_feature_frame(paths, config)
    X, y = data.feature_target_split(df, config)
    results = modeling.cross_validate_registry(X, y, config)
    assert {"model", "test_accuracy_mean", "test_f1_macro_mean"}.issubset(results.columns)
    assert len(results) >= 3


def test_holdout_evaluation_prefers_non_trivial_models() -> None:
    """Holdout evaluations should rank advanced models better than dummy baselines."""

    config = ExperimentConfig()
    paths = DataPaths()
    df = data.cached_feature_frame(paths, config)
    X, y = data.feature_target_split(df, config)
    results = modeling.holdout_evaluation(X, y, config)
    assert len(results) == 4
    best = max(results, key=lambda item: item.test_accuracy)
    assert best.model_name in {"log_reg", "random_forest", "hist_gb"}

