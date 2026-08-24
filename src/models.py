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

from . import CONFIGS_DIR, RANDOM_STATE
from .preprocessing import build_preprocessor, validate_pre_accident_columns

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


def fit_multiclass(config: dict, X: pd.DataFrame, y: np.ndarray) -> Any:
    validate_pre_accident_columns(X)
    model = make_multiclass_pipeline(config)
    fit_parameters = balanced_fit_parameters(y) if config["balance"] == "weighted" else {}
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
