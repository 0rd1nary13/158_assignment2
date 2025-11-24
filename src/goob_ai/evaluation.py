"""Evaluation utilities for the Assignment 2 pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class EvaluationArtifacts:
    """Container for diagnostic outputs used in the notebook."""

    confusion: pd.DataFrame
    report_text: str


def build_confusion_matrix(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    labels: Sequence[str],
) -> pd.DataFrame:
    """Return a labeled confusion matrix dataframe."""

    predictions = estimator.predict(X)
    matrix = confusion_matrix(y, predictions, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def capture_evaluation_artifacts(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    labels: Sequence[str],
) -> EvaluationArtifacts:
    """Return structured evaluation outputs for downstream reporting."""

    confusion_df = build_confusion_matrix(estimator, X, y, labels)
    report = classification_report(y, estimator.predict(X), labels=labels)
    return EvaluationArtifacts(confusion=confusion_df, report_text=report)


def plot_confusion_matrix(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    labels: Sequence[str],
) -> ConfusionMatrixDisplay:
    """Plot and return a confusion matrix display."""

    disp = ConfusionMatrixDisplay(
        confusion_matrix(y, estimator.predict(X), labels=labels),
        display_labels=labels,
    )
    disp.plot(cmap="Blues")
    return disp

