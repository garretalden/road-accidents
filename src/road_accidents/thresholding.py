"""Fatal-class threshold rules and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .config import CLASS_NAMES


def apply_fatal_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Predict Fatal above ``threshold``; otherwise choose Serious vs Slight."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Expected probabilities with shape (n, {len(CLASS_NAMES)})")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    # np.argmax resolves an exact Serious/Slight tie in favor of Serious.
    nonfatal = np.argmax(probabilities[:, 1:], axis=1) + 1
    return np.where(probabilities[:, 0] >= threshold, 0, nonfatal)


def threshold_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = apply_fatal_threshold(probabilities, threshold)
    return {
        "threshold": float(threshold),
        "fatal_precision": float(
            precision_score(y_true, predictions, labels=[0], average=None, zero_division=0)[0]
        ),
        "fatal_recall": float(
            recall_score(y_true, predictions, labels=[0], average=None, zero_division=0)[0]
        ),
        "fatal_f1": float(
            f1_score(y_true, predictions, labels=[0], average=None, zero_division=0)[0]
        ),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "predicted_fatal_proportion": float(np.mean(predictions == 0)),
    }


def evaluate_thresholds(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: list[float] | np.ndarray
) -> list[dict[str, float]]:
    return [threshold_metrics(y_true, probabilities, float(value)) for value in thresholds]


def select_macro_f1_threshold(results: list[dict[str, float]]) -> dict[str, float]:
    """Select by macro F1 with deterministic, conservative tie-breakers."""
    return min(
        results,
        key=lambda result: (
            -result["macro_f1"],
            -result["fatal_f1"],
            result["predicted_fatal_proportion"],
            -result["threshold"],
        ),
    )


def select_fatal_f1_threshold(results: list[dict[str, float]]) -> dict[str, float]:
    return min(
        results,
        key=lambda result: (
            -result["fatal_f1"],
            -result["macro_f1"],
            -result["fatal_precision"],
            result["predicted_fatal_proportion"],
            -result["threshold"],
        ),
    )


def search_fatal_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    coarse_step: float = 0.01,
    refinement_radius: float = 0.01,
    refinement_step: float = 0.0005,
) -> dict:
    """Run a broad threshold sweep, then refine around both validation optima."""
    coarse = np.arange(0.0, 1.0 + coarse_step / 2, coarse_step)
    coarse_results = evaluate_thresholds(y_true, probabilities, coarse)
    anchors = {
        select_macro_f1_threshold(coarse_results)["threshold"],
        select_fatal_f1_threshold(coarse_results)["threshold"],
    }
    thresholds = {round(float(value), 6) for value in coarse}
    for anchor in anchors:
        start = max(0.0, anchor - refinement_radius)
        stop = min(1.0, anchor + refinement_radius)
        thresholds.update(
            round(float(value), 6)
            for value in np.arange(start, stop + refinement_step / 2, refinement_step)
        )

    results = evaluate_thresholds(y_true, probabilities, sorted(thresholds))
    return {
        "grid": {
            "coarse_step": coarse_step,
            "refinement_radius": refinement_radius,
            "refinement_step": refinement_step,
            "threshold_count": len(results),
        },
        "selected_for_macro_f1": select_macro_f1_threshold(results),
        "selected_for_fatal_f1": select_fatal_f1_threshold(results),
        "thresholds": results,
    }
