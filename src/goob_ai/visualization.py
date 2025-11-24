"""Reusable visualization helpers for the notebook."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_target_distribution(df: pd.DataFrame, target_column: str) -> plt.Axes:
    """Plot the raw distribution of the discrete target variable."""

    ax = sns.countplot(data=df, x=target_column, palette="viridis")
    ax.set_title("Wine quality distribution")
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Count")
    return ax


def plot_feature_vs_target(
    df: pd.DataFrame,
    feature: str,
    target_column: str,
) -> plt.Axes:
    """Plot boxplots for a single feature vs. the target variable."""

    ax = sns.boxplot(data=df, x=target_column, y=feature, palette="magma")
    ax.set_title(f"{feature} vs quality")
    return ax


def plot_correlation_heatmap(df: pd.DataFrame, columns: Iterable[str]) -> plt.Axes:
    """Visualize correlations for the provided feature subset."""

    corr = df[list(columns)].corr(numeric_only=True)
    ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    ax.set_title("Feature correlation heatmap")
    return ax

