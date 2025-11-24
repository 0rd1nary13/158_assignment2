"""Centralized configuration objects for the Assignment 2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataPaths:
    """Represents canonical on-disk locations used throughout the project."""

    root_dir: Path = PROJECT_ROOT
    raw_subdir: Path = Path("data") / "raw"
    processed_subdir: Path = Path("data") / "processed"
    reports_subdir: Path = Path("reports")

    @property
    def raw_dir(self) -> Path:
        """Return the directory that stores immutable raw assets."""

        return self.root_dir / self.raw_subdir

    @property
    def processed_dir(self) -> Path:
        """Return the directory that stores derived datasets."""

        return self.root_dir / self.processed_subdir

    @property
    def reports_dir(self) -> Path:
        """Return the directory that stores generated figures and HTML files."""

        return self.root_dir / self.reports_subdir

    @property
    def raw_wine_quality(self) -> Path:
        """Return the local path for the Wine Quality dataset."""

        return self.raw_dir / "winequality-red.csv"

    @property
    def feature_cache(self) -> Path:
        """Return the location of the cached feature matrix."""

        return self.processed_dir / "wine_features.parquet"


@dataclass(frozen=True)
class ExperimentConfig:
    """Holds tunable knobs for modeling and evaluation."""

    score_column: str = "quality"
    target_column: str = "quality_label"
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    scoring_metrics: Sequence[str] = field(
        default_factory=lambda: ("accuracy", "f1_macro", "balanced_accuracy"),
    )

    @property
    def seed_sequence(self) -> list[int]:
        """Return deterministic seeds for repeated experiments."""

        return [self.random_state + offset for offset in range(self.n_splits)]

