"""Leakage-safe cross-validation for fixed accident-severity models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import CLASS_NAMES, DOWNSAMPLE_TARGETS, RANDOM_STATE
from .evaluate import evaluate, evaluate_predictions
from .preprocessing import build_preprocessor


BalanceStrategy = Literal["weighted", "downsampled"]


@dataclass(frozen=True)
class ValidationSpec:
    """Fixed estimator and its existing class-balancing strategy."""

    name: str
    slug: str
    balance: BalanceStrategy
    estimator_factory: Callable[[], BaseEstimator]
    hyperparameters: dict[str, Any]


def _class_counts(y: np.ndarray) -> dict[str, int]:
    counts = pd.Series(y).value_counts()
    return {CLASS_NAMES[i]: int(counts.get(i, 0)) for i in range(len(CLASS_NAMES))}


def _training_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    balance: BalanceStrategy,
    downsample_targets: dict[int, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    if balance == "weighted":
        return X, y

    counts = pd.Series(y).value_counts()
    strategy = {0: int(counts[0]), **downsample_targets}
    sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=RANDOM_STATE)
    return sampler.fit_resample(X, y)


def _fit(
    spec: ValidationSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    downsample_targets: dict[int, int],
) -> tuple[Pipeline, int, dict[str, int]]:
    X_fit, y_fit = _training_rows(X, y, spec.balance, downsample_targets)
    preprocessor = build_preprocessor()
    X_encoded = preprocessor.fit_transform(X_fit)
    estimator = spec.estimator_factory()

    if spec.balance == "weighted":
        from sklearn.utils.class_weight import compute_sample_weight

        estimator.fit(X_encoded, y_fit, sample_weight=compute_sample_weight("balanced", y_fit))
    else:
        estimator.fit(X_encoded, y_fit)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    return pipeline, len(y_fit), _class_counts(y_fit)


def _summary(folds: list[dict]) -> dict:
    def stats(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }

    per_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        per_class[class_name] = {
            metric: stats([fold[f"per_class_{metric}"][class_index] for fold in folds])
            for metric in ("precision", "recall", "f1")
        }
    return {
        "macro_f1": stats([fold["macro_f1"] for fold in folds]),
        "per_class": per_class,
    }


def validate_fixed_model(
    spec: ValidationSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    n_splits: int = 5,
    downsample_targets: dict[int, int] = DOWNSAMPLE_TARGETS,
    on_fold: Callable[[int, dict], None] | None = None,
    on_final_test: Callable[[], None] | None = None,
) -> tuple[dict, Pipeline]:
    """Cross-validate on training only, then fit once and score the final test set."""
    cv = cross_validate_fixed_model(
        spec,
        X_train,
        y_train,
        n_splits=n_splits,
        downsample_targets=downsample_targets,
        on_fold=on_fold,
    )
    final, final_pipeline = fit_final_model(
        spec,
        X_train,
        y_train,
        X_test,
        y_test,
        downsample_targets=downsample_targets,
        on_final_test=on_final_test,
    )
    return {**cv, **final}, final_pipeline


def cross_validate_fixed_model(
    spec: ValidationSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_splits: int = 5,
    downsample_targets: dict[int, int] = DOWNSAMPLE_TARGETS,
    on_fold: Callable[[int, dict], None] | None = None,
) -> dict:
    """Evaluate a fixed model using only stratified folds of the training split."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = []

    for fold_number, (fit_indices, validation_indices) in enumerate(
        splitter.split(X_train, y_train), start=1
    ):
        X_fit = X_train.iloc[fit_indices]
        y_fit = y_train[fit_indices]
        X_validation = X_train.iloc[validation_indices]
        y_validation = y_train[validation_indices]

        pipeline, fitted_rows, fitted_counts = _fit(
            spec, X_fit, y_fit, downsample_targets
        )
        metrics = evaluate_predictions(
            y_validation, pipeline.predict(X_validation), spec.name
        )
        metrics.update(
            {
                "fold": fold_number,
                "training_rows": int(len(fit_indices)),
                "fitted_rows": int(fitted_rows),
                "validation_rows": int(len(validation_indices)),
                "training_class_counts": _class_counts(y_fit),
                "fitted_class_counts": fitted_counts,
                "validation_class_counts": _class_counts(y_validation),
            }
        )
        folds.append(metrics)
        if on_fold is not None:
            on_fold(fold_number, metrics)

    return {
        "name": spec.name,
        "slug": spec.slug,
        "balance": spec.balance,
        "hyperparameters": spec.hyperparameters,
        "cv": {"folds": folds, "summary": _summary(folds)},
    }


def fit_final_model(
    spec: ValidationSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    downsample_targets: dict[int, int] = DOWNSAMPLE_TARGETS,
    on_final_test: Callable[[], None] | None = None,
) -> tuple[dict, Pipeline]:
    """Fit on all training data and evaluate once on the final test split."""
    final_pipeline, fitted_rows, fitted_counts = _fit(
        spec, X_train, y_train, downsample_targets
    )
    if on_final_test is not None:
        on_final_test()
    test_metrics = evaluate(final_pipeline, X_test, y_test, spec.name)

    result = {
        "final_fit": {
            "training_rows": int(len(y_train)),
            "fitted_rows": int(fitted_rows),
            "training_class_counts": _class_counts(y_train),
            "fitted_class_counts": fitted_counts,
        },
        "test": test_metrics,
    }
    return result, final_pipeline
