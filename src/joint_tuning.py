"""Sampling, selection, and reporting helpers for joint XGBoost tuning."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import loguniform, uniform
from sklearn.model_selection import ParameterSampler

from . import RANDOM_STATE
from .evaluation import summarize_folds
from .models import interpolated_fit_parameters


def _parameter_distribution(specification: dict):
    if "values" in specification:
        return specification["values"]
    low = float(specification["low"])
    high = float(specification["high"])
    if not high > low:
        raise ValueError("search-space upper bound must exceed lower bound")
    if specification["distribution"] == "uniform":
        return uniform(loc=low, scale=high - low)
    if specification["distribution"] == "loguniform":
        if low <= 0:
            raise ValueError("log-uniform lower bound must be positive")
        return loguniform(low, high)
    raise ValueError(f"unsupported distribution: {specification['distribution']}")


def sample_joint_candidates(config: dict) -> list[dict]:
    """Sample the configured joint search space reproducibly."""
    distributions = {
        name: _parameter_distribution(specification)
        for name, specification in config["search_space"].items()
    }
    return list(
        ParameterSampler(
            distributions,
            n_iter=int(config["candidate_count"]),
            random_state=RANDOM_STATE,
        )
    )


def fit_parameters_for_alpha(alpha: float) -> Callable[[np.ndarray], dict[str, np.ndarray]]:
    """Bind one candidate's alpha to fold-local sample-weight computation."""
    value = float(alpha)

    def fit_parameters(fold_y: np.ndarray) -> dict[str, np.ndarray]:
        return interpolated_fit_parameters(fold_y, value)

    return fit_parameters


def summarize_joint_candidate(
    candidate_number: int,
    alpha: float,
    parameters: dict,
    folds: list[dict],
) -> dict:
    """Flatten the required search metrics into one auditable candidate record."""
    summary = summarize_folds(folds)
    per_class = summary["per_class"]
    return {
        "candidate": int(candidate_number),
        "alpha": float(alpha),
        **parameters,
        "macro_f1_mean": summary["macro_f1"]["mean"],
        "macro_f1_std": summary["macro_f1"]["std"],
        "fatal_precision_mean": per_class["Fatal"]["precision"]["mean"],
        "fatal_recall_mean": per_class["Fatal"]["recall"]["mean"],
        "fatal_f1_mean": per_class["Fatal"]["f1"]["mean"],
        "serious_f1_mean": per_class["Serious"]["f1"]["mean"],
        "slight_f1_mean": per_class["Slight"]["f1"]["mean"],
    }


def select_joint_candidate(rows: list[dict]) -> dict:
    """Select maximum mean macro-F1, retaining sample order on an exact tie."""
    if not rows:
        raise ValueError("at least one joint-tuning result is required")
    return max(rows, key=lambda row: (row["macro_f1_mean"], -row["candidate"]))


def render_joint_tuning_markdown(report: dict) -> str:
    """Render the search, validation, and optional held-out result as Markdown."""
    winner = report["selected_candidate"]
    validation = report["validation"]
    lines = [
        "# Joint XGBoost and class-weight tuning",
        "",
        f"- Selection data: {report['selection_data']}",
        f"- Search: {report['candidate_count']} candidates × {report['search_folds']} folds",
        f"- Selected candidate: {winner['candidate']}",
        f"- Selected alpha: {winner['alpha']:.6g}",
        f"- Search macro-F1: {winner['macro_f1_mean']:.6f} ± {winner['macro_f1_std']:.6f}",
        f"- Five-fold validation macro-F1: "
        f"{validation['macro_f1']['mean']:.6f} ± {validation['macro_f1']['std']:.6f}",
        "",
        "## Sampled candidates",
        "",
    ]
    columns = [
        "candidate", "alpha", "max_depth", "learning_rate", "min_child_weight",
        "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
        "n_estimators", "macro_f1_mean", "macro_f1_std", "serious_f1_mean",
        "fatal_precision_mean", "fatal_recall_mean", "fatal_f1_mean", "slight_f1_mean",
    ]
    lines.extend([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ])
    for row in report["search"]:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    if "untouched_test" in report:
        result = report["untouched_test"]
        lines.extend([
            "",
            "## Untouched test result",
            "",
            f"- Accuracy: {result['accuracy']:.6f}",
            f"- Macro-F1: {result['macro_f1']:.6f}",
            f"- Fatal precision: {result['per_class_precision'][0]:.6f}",
            f"- Fatal recall: {result['per_class_recall'][0]:.6f}",
            f"- Fatal F1: {result['per_class_f1'][0]:.6f}",
            f"- Serious F1: {result['per_class_f1'][1]:.6f}",
            f"- Slight F1: {result['per_class_f1'][2]:.6f}",
        ])
    return "\n".join(lines) + "\n"
