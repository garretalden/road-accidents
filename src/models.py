"""Model factories and ordinal probability composition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import CONFIGS_DIR, PROJECT_ROOT, RANDOM_STATE
from .preprocessing import build_preprocessor, validate_pre_accident_columns
from .weighting import interpolated_sample_weight

BASE_XGB_PARAMETERS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


class DownsampleStrategy:
    """Pickle-safe callable that retains Fatal rows and caps majority classes."""

    def __init__(self, targets: dict[str, int]):
        self.targets = targets

    def __call__(self, y: np.ndarray) -> dict[int, int]:
        counts = pd.Series(y).value_counts()
        return {
            0: int(counts.get(0, 0)),
            1: min(int(counts.get(1, 0)), int(self.targets["Serious"])),
            2: min(int(counts.get(2, 0)), int(self.targets["Slight"])),
        }


def load_config(name_or_path: str | Path) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIGS_DIR / f"{path}.json"
    return json.loads(path.read_text())


def load_selected_tuned_parameters(path: str | Path) -> dict:
    """Load and validate the parameter winner produced by tuned-model search."""
    result_path = Path(path)
    if not result_path.is_absolute():
        result_path = PROJECT_ROOT / result_path
    if not result_path.exists():
        raise FileNotFoundError(
            f"Tuned-parameter results not found at {result_path}. "
            "Run `make train-tuned` first."
        )
    try:
        report = json.loads(result_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Tuned-parameter results are not valid JSON: {result_path}") from error
    parameters = report.get("selected_parameters")
    required = {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    }
    if not isinstance(parameters, dict):
        raise ValueError(f"{result_path} does not contain selected_parameters")
    missing = sorted(required.difference(parameters))
    if missing:
        raise ValueError(f"selected_parameters is missing required values: {missing}")
    return parameters.copy()


def make_multiclass_pipeline(config: dict) -> Pipeline:
    """Build a self-contained raw-feature XGBoost pipeline."""
    parameters = {**BASE_XGB_PARAMETERS, **config["parameters"]}
    steps = [("preprocessor", build_preprocessor())]
    if config["balance"] == "downsampled":
        steps.append(
            (
                "sampler",
                RandomUnderSampler(
                    sampling_strategy=DownsampleStrategy(config["downsample_targets"]),
                    random_state=RANDOM_STATE,
                ),
            )
        )
        return ImbalancedPipeline([*steps, ("model", XGBClassifier(**parameters))])
    return Pipeline([*steps, ("model", XGBClassifier(**parameters))])


def balanced_fit_parameters(y: np.ndarray) -> dict[str, np.ndarray]:
    return {"model__sample_weight": compute_sample_weight("balanced", y)}


def interpolated_fit_parameters(y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    return {"model__sample_weight": interpolated_sample_weight(y, alpha)}


def fit_multiclass(config: dict, X: pd.DataFrame, y: np.ndarray) -> Any:
    validate_pre_accident_columns(X)
    model = make_multiclass_pipeline(config)
    if config["balance"] == "weighted":
        fit_parameters = balanced_fit_parameters(y)
    elif config["balance"] == "interpolated":
        fit_parameters = interpolated_fit_parameters(y, config["weight_alpha"])
    else:
        fit_parameters = {}
    return model.fit(X, y, **fit_parameters)


def cumulative_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(y)
    return np.isin(values, [0, 1]).astype(np.uint8), (values == 0).astype(np.uint8)


def make_binary_pipeline(parameters: dict) -> Pipeline:
    binary_parameters = {
        key: value for key, value in {**BASE_XGB_PARAMETERS, **parameters}.items()
        if key not in {"num_class", "objective"}
    }
    binary_parameters["objective"] = "binary:logistic"
    return Pipeline(
        [("preprocessor", build_preprocessor()), ("model", XGBClassifier(**binary_parameters))]
    )


def fit_ordinal_models(
    config: dict, X: pd.DataFrame, y: np.ndarray
) -> tuple[Pipeline, Pipeline]:
    validate_pre_accident_columns(X)
    serious_target, fatal_target = cumulative_targets(y)
    serious = make_binary_pipeline(config["parameters"])
    fatal = make_binary_pipeline(config["parameters"])
    serious.fit(X, serious_target, model__sample_weight=compute_sample_weight("balanced", serious_target))
    fatal.fit(X, fatal_target, model__sample_weight=compute_sample_weight("balanced", fatal_target))
    return serious, fatal


def ordinal_probabilities(
    serious_or_worse_model: Any, fatal_model: Any, X: pd.DataFrame
) -> np.ndarray:
    serious = serious_or_worse_model.predict_proba(X)[:, 1]
    fatal = fatal_model.predict_proba(X)[:, 1]
    violation = fatal > serious
    midpoint = (fatal + serious) / 2
    serious = np.where(violation, midpoint, serious)
    fatal = np.where(violation, midpoint, fatal)
    probabilities = np.column_stack([fatal, serious - fatal, 1 - serious])
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise RuntimeError("ordinal class probabilities must sum to one")
    return probabilities


class OrdinalPredictor:
    """Prediction adapter around the two separately persisted ordinal pipelines."""

    def __init__(self, serious_or_worse_model: Any, fatal_model: Any):
        self.serious_or_worse_model = serious_or_worse_model
        self.fatal_model = fatal_model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return ordinal_probabilities(self.serious_or_worse_model, self.fatal_model, X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
