"""Leakage-safe model evaluation, threshold selection, and error cohorts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from . import CLASS_NAMES, RANDOM_STATE


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """Compute the project's standard multiclass metrics."""
    labels = range(len(CLASS_NAMES))
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_precision": precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist(),
        "per_class_f1": f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate(model: Any, X_test: pd.DataFrame, y_test: np.ndarray, name: str) -> dict:
    return evaluate_predictions(y_test, np.asarray(model.predict(X_test)), name)


def apply_fatal_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Predict Fatal above threshold, otherwise choose the likelier non-Fatal class."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES):
        raise ValueError("probabilities must have shape (rows, 3)")
    predictions = np.argmax(values[:, 1:], axis=1) + 1
    predictions[values[:, 0] >= threshold] = 0
    return predictions


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = apply_fatal_threshold(probabilities, threshold)
    metrics = evaluate_predictions(y_true, predictions, "threshold candidate")
    return {
        "threshold": float(threshold),
        "macro_f1": metrics["macro_f1"],
        "fatal_precision": metrics["per_class_precision"][0],
        "fatal_recall": metrics["per_class_recall"][0],
        "fatal_f1": metrics["per_class_f1"][0],
        "predicted_fatal_proportion": float(np.mean(predictions == 0)),
    }


def select_fatal_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, *, grid_size: int = 201
) -> tuple[dict, list[dict]]:
    """Select macro-F1 threshold, breaking ties by Fatal F1 then higher threshold."""
    rows = [threshold_metrics(y_true, probabilities, value) for value in np.linspace(0, 1, grid_size)]
    selected = max(rows, key=lambda row: (row["macro_f1"], row["fatal_f1"], row["threshold"]))
    return selected, rows


def cross_validate_pipeline(
    pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    name: str,
    n_splits: int = 5,
    fit_parameters: Callable[[np.ndarray], dict] | None = None,
    collect_probabilities: bool = False,
) -> tuple[list[dict], np.ndarray | None]:
    """Fit a fresh full pipeline per fold and optionally return OOF probabilities."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds: list[dict] = []
    oof = np.full((len(y), len(CLASS_NAMES)), np.nan) if collect_probabilities else None
    for number, (fit_index, validation_index) in enumerate(splitter.split(X, y), start=1):
        fold_model = clone(pipeline)
        params = fit_parameters(y[fit_index]) if fit_parameters else {}
        fold_model.fit(X.iloc[fit_index], y[fit_index], **params)
        prediction = fold_model.predict(X.iloc[validation_index])
        metrics = evaluate_predictions(y[validation_index], prediction, name)
        metrics.update(
            {"fold": number, "training_rows": len(fit_index), "validation_rows": len(validation_index)}
        )
        folds.append(metrics)
        if oof is not None:
            oof[validation_index] = fold_model.predict_proba(X.iloc[validation_index])
    if oof is not None and not np.isfinite(oof).all():
        raise RuntimeError("every training row must receive one finite OOF probability vector")
    return folds, oof


def summarize_folds(folds: list[dict]) -> dict:
    """Summarize fold metrics with sample standard deviations."""
    def stats(values: list[float]) -> dict[str, float]:
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}

    return {
        "macro_f1": stats([fold["macro_f1"] for fold in folds]),
        "accuracy": stats([fold["accuracy"] for fold in folds]),
        "per_class": {
            name: {
                metric: stats([fold[f"per_class_{metric}"][index] for fold in folds])
                for metric in ("precision", "recall", "f1")
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def upsert_cv_results(name: str, slug: str, folds: list[dict], path: Path) -> None:
    """Upsert one compact row per validation fold into a shared CSV."""
    rows = []
    for fold in folds:
        rows.append(
            {
                "model": name,
                "slug": slug,
                "fold": fold["fold"],
                "macro_f1": fold["macro_f1"],
                "accuracy": fold["accuracy"],
                **{
                    f"{class_name.lower()}_{metric}": fold[f"per_class_{metric}"][index]
                    for index, class_name in enumerate(CLASS_NAMES)
                    for metric in ("precision", "recall", "f1")
                },
            }
        )
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not current.empty:
        current = current[current["slug"] != slug]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([current, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def build_error_cohorts(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if true.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    labels = np.asarray(CLASS_NAMES, dtype=object)
    correct = true == predicted
    return pd.DataFrame(
        {
            "true_label": labels[true],
            "predicted_label": labels[predicted],
            "is_correct": correct,
            "error_type": np.where(correct, "Correct", labels[true] + " → " + labels[predicted]),
        }
    )


def histogram_overlap(left: pd.Series, right: pd.Series, *, bins: int = 20) -> float:
    a = pd.to_numeric(left, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(right, errors="coerce").dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if min(a.min(), b.min()) == max(a.max(), b.max()):
        return 1.0
    edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
    ah, _ = np.histogram(a, edges)
    bh, _ = np.histogram(b, edges)
    return float(np.minimum(ah / ah.sum(), bh / bh.sum()).sum())
