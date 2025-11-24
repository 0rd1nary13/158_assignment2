"""Data access helpers for Assignment 2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd
import requests

from goob_ai.config import DataPaths, ExperimentConfig

logger = logging.getLogger(__name__)

WINE_DATA_URL: Final[str] = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)


def ensure_directories(paths: DataPaths) -> None:
    """Create required directories if they do not already exist."""

    for directory in (paths.raw_dir, paths.processed_dir, paths.reports_dir):
        directory.mkdir(parents=True, exist_ok=True)


def download_wine_quality(paths: DataPaths, *, force: bool = False) -> Path:
    """Download the Wine Quality dataset locally if needed.

    Args:
        paths: Canonical path container.
        force: Whether to re-download even if the file already exists.

    Returns:
        The path to the downloaded CSV file.
    """

    ensure_directories(paths)
    target_path = paths.raw_wine_quality
    if target_path.exists() and not force:
        logger.info("Wine Quality dataset already present at %s", target_path)
        return target_path

    logger.info("Downloading Wine Quality dataset from %s", WINE_DATA_URL)
    response = requests.get(WINE_DATA_URL, timeout=60)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    logger.info("Saved dataset to %s (%.2f KB)", target_path, target_path.stat().st_size / 1024)
    return target_path


def load_wine_quality(paths: DataPaths) -> pd.DataFrame:
    """Load the Wine Quality dataset into a DataFrame."""

    csv_path = download_wine_quality(paths)
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [column.strip().replace(" ", "_") for column in df.columns]
    logger.debug("Loaded dataframe with shape: %s", df.shape)
    return df


def add_quality_label(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Add a categorical quality label suitable for classification."""

    label_bins = [-float("inf"), 5, 6, float("inf")]
    label_names = ["low", "medium", "high"]
    labelled_df = df.copy()
    labelled_df[config.target_column] = pd.cut(
        labelled_df[config.score_column],
        bins=label_bins,
        labels=label_names,
    )
    return labelled_df


def cached_feature_frame(paths: DataPaths, config: ExperimentConfig) -> pd.DataFrame:
    """Return a feature matrix, saving a cached Parquet file for re-use."""

    cache_path = paths.feature_cache
    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        return pd.read_parquet(cache_path)

    df = load_wine_quality(paths)
    df = add_quality_label(df, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached features at %s", cache_path)
    return df


def feature_target_split(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features and the classification target."""

    drop_columns = {config.target_column, config.score_column}
    features = df.drop(columns=list(drop_columns))
    target = df[config.target_column]
    return features, target


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enriched descriptive statistics for reporting."""

    stats = df.describe().T
    stats["coefficient_of_variation"] = stats["std"] / stats["mean"].replace(0, pd.NA)
    return stats.reset_index(names="feature")

