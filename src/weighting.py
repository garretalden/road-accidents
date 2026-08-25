"""Class-weight interpolation and deterministic one-dimensional search helpers."""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


def interpolated_sample_weight(y: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly interpolate between unit and balanced sample weights."""
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("weight alpha must be finite and between 0 and 1")
    balanced = compute_sample_weight("balanced", y)
    return np.ones_like(balanced, dtype=float) + value * (balanced - 1.0)


def fine_alpha_grid(center: float, *, radius: float = 0.15, step: float = 0.05) -> list[float]:
    """Build an inclusive, clipped alpha grid without floating-point drift."""
    center_units = round(float(center) * 1000)
    radius_units = round(float(radius) * 1000)
    step_units = round(float(step) * 1000)
    if step_units <= 0 or radius_units < 0:
        raise ValueError("fine-grid radius must be nonnegative and step must be positive")
    if not 0 <= center_units <= 1000:
        raise ValueError("fine-grid center must be between 0 and 1")
    lower = max(0, center_units - radius_units)
    upper = min(1000, center_units + radius_units)
    return [value / 1000 for value in range(lower, upper + 1, step_units)]


def select_alpha_result(rows: list[dict]) -> dict:
    """Select maximum mean macro-F1, breaking exact ties toward lower alpha."""
    if not rows:
        raise ValueError("at least one alpha result is required")
    return max(rows, key=lambda row: (row["macro_f1_mean"], -row["alpha"]))
