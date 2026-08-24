"""Cumulative-binary ordinal XGBoost for accident severity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .evaluate import evaluate_predictions
from .preprocessing import build_preprocessor
from .validation import _class_counts, _summary

NAME = "XGBoost (ordinal cumulative, class-weighted)"
SLUG = "xgb_ordinal_weighted"

BINARY_BASE_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def cumulative_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return targets for Y >= Serious and Y == Fatal in encoded class order."""
    y = np.asarray(y)
    return np.isin(y, [0, 1]).astype(np.uint8), (y == 0).astype(np.uint8)


def enforce_cumulative_order(
    at_least_serious: np.ndarray, fatal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project cumulative probabilities onto P(Fatal) <= P(Y >= Serious)."""
    at_least_serious = np.asarray(at_least_serious, dtype=float)
    fatal = np.asarray(fatal, dtype=float)
    if at_least_serious.shape != fatal.shape:
        raise ValueError("Cumulative probability arrays must have the same shape")
    if not (
        np.isfinite(at_least_serious).all()
        and np.isfinite(fatal).all()
        and np.all((0 <= at_least_serious) & (at_least_serious <= 1))
        and np.all((0 <= fatal) & (fatal <= 1))
    ):
        raise ValueError("Cumulative probabilities must be finite and between zero and one")

    violation = fatal > at_least_serious
    midpoint = (at_least_serious + fatal) / 2
    return (
        np.where(violation, midpoint, at_least_serious),
        np.where(violation, midpoint, fatal),
    )


def cumulative_to_class_probabilities(
    at_least_serious: np.ndarray, fatal: np.ndarray
) -> np.ndarray:
    """Convert monotone cumulative outputs to Fatal/Serious/Slight probabilities."""
    at_least_serious, fatal = enforce_cumulative_order(at_least_serious, fatal)
    probabilities = np.column_stack(
        [fatal, at_least_serious - fatal, 1.0 - at_least_serious]
    )
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Ordinal class probabilities must sum to one")
    if np.any(probabilities < -1e-12):
        raise RuntimeError("Ordinal class probabilities must be nonnegative")
    return probabilities


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.uint8)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "positive_proportion": float(np.mean(prediction == 1)),
    }


def _binary_summary(folds: list[dict], task: str) -> dict:
    def stats(metric: str) -> dict[str, float]:
        values = [fold["binary_tasks"][task][metric] for fold in folds]
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}

    return {metric: stats(metric) for metric in ("accuracy", "precision", "recall", "f1")}


@dataclass
class OrdinalXGBPipeline:
    """Self-contained raw-feature pipeline with two cumulative binary models."""

    preprocessor: Any
    at_least_serious_model: Any
    fatal_model: Any

    def predict_cumulative_proba(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.preprocessor.transform(X)
        return (
            self.at_least_serious_model.predict_proba(encoded)[:, 1],
            self.fatal_model.predict_proba(encoded)[:, 1],
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        at_least_serious, fatal = self.predict_cumulative_proba(X)
        return cumulative_to_class_probabilities(at_least_serious, fatal)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def _fit_ordinal(
    X: pd.DataFrame,
    y: np.ndarray,
    parameters: dict[str, Any],
    *,
    estimator_factory: Callable[..., Any] = XGBClassifier,
) -> tuple[OrdinalXGBPipeline, dict]:
    preprocessor = build_preprocessor()
    encoded = preprocessor.fit_transform(X)
    at_least_serious_target, fatal_target = cumulative_targets(y)
    model_parameters = {**BINARY_BASE_PARAMS, **parameters}

    at_least_serious_model = estimator_factory(**model_parameters)
    at_least_serious_model.fit(
        encoded,
        at_least_serious_target,
        sample_weight=compute_sample_weight("balanced", at_least_serious_target),
    )
    fatal_model = estimator_factory(**model_parameters)
    fatal_model.fit(
        encoded,
        fatal_target,
        sample_weight=compute_sample_weight("balanced", fatal_target),
    )
    metadata = {
        "training_rows": int(len(y)),
        "training_class_counts": _class_counts(y),
        "binary_training_counts": {
            "at_least_serious": {
                "negative": int(np.sum(at_least_serious_target == 0)),
                "positive": int(np.sum(at_least_serious_target == 1)),
            },
            "fatal": {
                "negative": int(np.sum(fatal_target == 0)),
                "positive": int(np.sum(fatal_target == 1)),
            },
        },
    }
    return OrdinalXGBPipeline(preprocessor, at_least_serious_model, fatal_model), metadata


def cross_validate_ordinal_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    parameters: dict[str, Any],
    *,
    n_splits: int = 5,
    on_fold: Callable[[int, dict], None] | None = None,
) -> dict:
    """Validate both binary tasks and their combined ordinal prediction fold-safely."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fold_number, (fit_indices, validation_indices) in enumerate(
        splitter.split(X_train, y_train), start=1
    ):
        pipeline, fit_metadata = _fit_ordinal(
            X_train.iloc[fit_indices], y_train[fit_indices], parameters
        )
        X_validation = X_train.iloc[validation_indices]
        y_validation = y_train[validation_indices]
        at_least_serious_probability, fatal_probability = pipeline.predict_cumulative_proba(
            X_validation
        )
        at_least_serious_target, fatal_target = cumulative_targets(y_validation)
        raw_violations = fatal_probability > at_least_serious_probability
        class_probabilities = cumulative_to_class_probabilities(
            at_least_serious_probability, fatal_probability
        )
        metrics = evaluate_predictions(
            y_validation, np.argmax(class_probabilities, axis=1), NAME
        )
        metrics.update(
            {
                "fold": fold_number,
                **fit_metadata,
                "validation_rows": int(len(validation_indices)),
                "validation_class_counts": _class_counts(y_validation),
                "binary_tasks": {
                    "at_least_serious": _binary_metrics(
                        at_least_serious_target, at_least_serious_probability
                    ),
                    "fatal": _binary_metrics(fatal_target, fatal_probability),
                },
                "ordering": {
                    "raw_violation_count": int(np.sum(raw_violations)),
                    "raw_violation_proportion": float(np.mean(raw_violations)),
                    "corrected_probability_count": int(np.sum(raw_violations)),
                },
            }
        )
        folds.append(metrics)
        if on_fold is not None:
            on_fold(fold_number, metrics)

    return {
        "name": NAME,
        "slug": SLUG,
        "balance": "weighted separately per binary task",
        "hyperparameters": {**BINARY_BASE_PARAMS, **parameters},
        "cv": {
            "folds": folds,
            "summary": _summary(folds),
            "binary_tasks": {
                task: _binary_summary(folds, task) for task in ("at_least_serious", "fatal")
            },
            "ordering": {
                "raw_violation_count": int(
                    sum(fold["ordering"]["raw_violation_count"] for fold in folds)
                ),
                "validation_rows": int(sum(fold["validation_rows"] for fold in folds)),
            },
        },
    }


def fit_final_ordinal_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    parameters: dict[str, Any],
    *,
    on_final_test: Callable[[], None] | None = None,
) -> tuple[dict, OrdinalXGBPipeline]:
    pipeline, fit_metadata = _fit_ordinal(X_train, y_train, parameters)
    if on_final_test is not None:
        on_final_test()
    at_least_serious_probability, fatal_probability = pipeline.predict_cumulative_proba(X_test)
    raw_violations = fatal_probability > at_least_serious_probability
    class_probabilities = cumulative_to_class_probabilities(
        at_least_serious_probability, fatal_probability
    )
    metrics = evaluate_predictions(y_test, np.argmax(class_probabilities, axis=1), NAME)
    at_least_serious_target, fatal_target = cumulative_targets(y_test)
    return (
        {
            "final_fit": fit_metadata,
            "test": metrics,
            "test_binary_tasks": {
                "at_least_serious": _binary_metrics(
                    at_least_serious_target, at_least_serious_probability
                ),
                "fatal": _binary_metrics(fatal_target, fatal_probability),
            },
            "test_ordering": {
                "raw_violation_count": int(np.sum(raw_violations)),
                "raw_violation_proportion": float(np.mean(raw_violations)),
                "post_correction_violation_count": 0,
            },
        },
        pipeline,
    )
