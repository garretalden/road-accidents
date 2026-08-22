"""Fold-safe randomized tuning for full-data, class-weighted XGBoost."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .evaluate import evaluate_predictions
from .preprocessing import build_preprocessor
from .validation import _summary

SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": loguniform(0.015, 0.15),
    "min_child_weight": [1, 2, 3, 5, 8, 13],
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": [0, 0.05, 0.1, 0.25, 0.5, 1, 2],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
    "reg_lambda": [0.25, 0.5, 1, 2, 5, 10, 20],
}

BASE_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def generate_candidates(n_iter: int = 12) -> list[dict[str, Any]]:
    """Generate a reproducible JSON-compatible randomized parameter sample."""
    sampled = ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=RANDOM_STATE)
    return [
        {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in candidate.items()
        }
        for candidate in sampled
    ]


def select_winner(results: list[dict]) -> dict:
    """Rank by macro F1, then Fatal F1, stability, and candidate number."""
    return min(
        results,
        key=lambda result: (
            -result["summary"]["macro_f1"]["mean"],
            -result["summary"]["per_class"]["Fatal"]["f1"]["mean"],
            result["summary"]["macro_f1"]["std"],
            result["candidate"],
        ),
    )


def tune_weighted_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    candidates: list[dict[str, Any]] | None = None,
    n_splits: int = 3,
    max_estimators: int = 600,
    early_stopping_rounds: int = 50,
    on_result: Callable[[int, int, dict], None] | None = None,
) -> dict:
    """Search candidates without allowing preprocessing or evaluation leakage."""
    candidates = candidates or generate_candidates()
    results = [
        {"candidate": number, "parameters": params, "folds": []}
        for number, params in enumerate(candidates, start=1)
    ]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fold_number, (fit_indices, validation_indices) in enumerate(
        splitter.split(X_train, y_train), start=1
    ):
        X_fit = X_train.iloc[fit_indices]
        y_fit = y_train[fit_indices]
        X_validation = X_train.iloc[validation_indices]
        y_validation = y_train[validation_indices]
        preprocessor = build_preprocessor()
        X_fit_encoded = preprocessor.fit_transform(X_fit)
        X_validation_encoded = preprocessor.transform(X_validation)
        weights = compute_sample_weight("balanced", y_fit)

        for result in results:
            model = XGBClassifier(
                **BASE_PARAMS,
                **result["parameters"],
                n_estimators=max_estimators,
                early_stopping_rounds=early_stopping_rounds,
            )
            model.fit(
                X_fit_encoded,
                y_fit,
                sample_weight=weights,
                eval_set=[(X_validation_encoded, y_validation)],
                verbose=False,
            )
            metrics = evaluate_predictions(
                y_validation, model.predict(X_validation_encoded), "candidate"
            )
            metrics.update(
                {
                    "fold": fold_number,
                    "best_iteration": int(model.best_iteration),
                    "selected_trees": int(model.best_iteration + 1),
                }
            )
            result["folds"].append(metrics)
            if on_result is not None:
                on_result(result["candidate"], fold_number, metrics)

    for result in results:
        result["summary"] = _summary(result["folds"])
        result["median_selected_trees"] = int(
            np.median([fold["selected_trees"] for fold in result["folds"]])
        )
    winner = select_winner(results)
    final_parameters = {**winner["parameters"], "n_estimators": winner["median_selected_trees"]}
    return {
        "n_splits": n_splits,
        "n_candidates": len(candidates),
        "max_estimators": max_estimators,
        "early_stopping_rounds": early_stopping_rounds,
        "primary_metric": "macro_f1",
        "candidates": results,
        "winner_candidate": winner["candidate"],
        "selected_parameters": final_parameters,
    }
