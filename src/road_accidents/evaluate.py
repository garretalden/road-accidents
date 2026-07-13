"""Evaluation metrics and Markdown results-table writer."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import CLASS_NAMES


def evaluate(model: Any, X_test: pd.DataFrame, y_test: np.ndarray, name: str) -> dict:
    """Compute macro + per-class metrics and the confusion matrix for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "per_class_precision": precision_score(
            y_test, y_pred, average=None, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_test, y_pred, average=None, zero_division=0
        ).tolist(),
        "per_class_f1": f1_score(y_test, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def write_results_table(results: list[dict], path: Path) -> None:
    """Write a Markdown table summarizing per-model macro-F1 and per-class F1."""
    lines = [
        "| Model | Macro F1 | Accuracy | F1 Fatal | F1 Serious | F1 Slight |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        f1s = r["per_class_f1"]
        lines.append(
            f"| {r['name']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | "
            f"{f1s[0]:.3f} | {f1s[1]:.3f} | {f1s[2]:.3f} |"
        )
    lines.append("")
    lines.append(f"Classes: {', '.join(f'{i}={n}' for i, n in enumerate(CLASS_NAMES))}")
    path.write_text("\n".join(lines) + "\n")
