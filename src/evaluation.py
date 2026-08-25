"""Leakage-safe model evaluation, threshold selection, and error cohorts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
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
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict], np.ndarray | None]:
    """Fit a fresh full pipeline per fold and optionally return OOF probabilities."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds: list[dict] = []
    oof = np.full((len(y), len(CLASS_NAMES)), np.nan) if collect_probabilities else None
    for number, (fit_index, validation_index) in enumerate(splitter.split(X, y), start=1):
        started = perf_counter()
        if progress_callback:
            progress_callback(f"fold {number}/{n_splits} started")
        fold_model = clone(pipeline)
        params = fit_parameters(y[fit_index]) if fit_parameters else {}
        fold_model.fit(X.iloc[fit_index], y[fit_index], **params)
        prediction = fold_model.predict(X.iloc[validation_index])
        metrics = evaluate_predictions(y[validation_index], prediction, name)
        metrics.update(
            {"fold": number, "training_rows": len(fit_index), "validation_rows": len(validation_index)}
        )
        folds.append(metrics)
        if progress_callback:
            progress_callback(
                f"fold {number}/{n_splits} completed: macro-F1={metrics['macro_f1']:.4f} "
                f"elapsed={perf_counter() - started:.1f}s"
            )
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


def aggregate_fatal_shap_by_source(
    preprocessor: Any,
    transformed_feature_names: list[str],
    fatal_shap_values: np.ndarray,
) -> list[dict]:
    """Aggregate transformed Fatal SHAP importance into interpretable source features."""
    categorical_columns = list(preprocessor.transformers_[0][2])
    numeric_columns = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.named_transformers_["cat"]
    source_columns: list[str] = []
    for index, (column, categories) in enumerate(zip(categorical_columns, encoder.categories_)):
        dropped = encoder.drop_idx_ is not None and encoder.drop_idx_[index] is not None
        source_columns.extend([column] * (len(categories) - int(dropped)))
    source_columns.extend(numeric_columns)
    if len(source_columns) != len(transformed_feature_names):
        raise ValueError("could not map every transformed feature to its source column")
    values = np.asarray(fatal_shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(source_columns):
        raise ValueError("Fatal SHAP values must have shape (rows, transformed features)")

    grouped: dict[str, dict] = {}
    for index, source in enumerate(source_columns):
        display_source = "Hour_of_Day" if source in {"hour_sin", "hour_cos"} else source
        entry = grouped.setdefault(
            display_source,
            {"source_feature": display_source, "mean_abs_fatal_shap": 0.0, "components": []},
        )
        entry["mean_abs_fatal_shap"] += float(np.mean(np.abs(values[:, index])))
        entry["components"].append(transformed_feature_names[index])
    return sorted(grouped.values(), key=lambda row: row["mean_abs_fatal_shap"], reverse=True)


def class_pair_overlaps(
    values: pd.Series,
    y_true: np.ndarray,
    *,
    categorical: bool,
    bins: int = 20,
) -> list[dict]:
    """Calculate probability-mass intersection for every severity-class pair."""
    labels = np.asarray(y_true, dtype=int)
    if len(values) != len(labels):
        raise ValueError("values and y_true must have equal lengths")
    distributions: dict[int, np.ndarray] = {}
    if categorical:
        cleaned = values.astype("string").fillna("<missing>")
        support = sorted(cleaned.unique().tolist())
        for class_index in range(len(CLASS_NAMES)):
            counts = cleaned[labels == class_index].value_counts(normalize=True)
            distributions[class_index] = counts.reindex(support, fill_value=0).to_numpy()
    else:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        finite = numeric[np.isfinite(numeric)]
        if len(finite) == 0:
            raise ValueError("numeric overlap requires at least one finite value")
        if float(finite.min()) == float(finite.max()):
            edges = np.array([finite.min() - 0.5, finite.max() + 0.5])
        else:
            edges = np.histogram_bin_edges(finite, bins=bins)
        for class_index in range(len(CLASS_NAMES)):
            class_values = numeric[(labels == class_index) & np.isfinite(numeric)]
            counts, _ = np.histogram(class_values, bins=edges)
            distributions[class_index] = counts / counts.sum() if counts.sum() else counts.astype(float)

    pairs = [(0, 1), (0, 2), (1, 2)]
    return [
        {
            "class_a": CLASS_NAMES[left],
            "class_b": CLASS_NAMES[right],
            "overlap": float(np.minimum(distributions[left], distributions[right]).sum()),
        }
        for left, right in pairs
    ]
