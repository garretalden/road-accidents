"""Fixed XGBoost definitions for leakage-safe weighted/baseline validation.

This is intentionally separate from ``xgb_class_weighted.py`` so that the
historical experiment remains frozen. There is no hyperparameter search here.
"""

from xgboost import XGBClassifier

from road_accidents.config import RANDOM_STATE
from road_accidents.validation import ValidationSpec


WEIGHTED_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "max_depth": 6,
    "min_child_weight": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

BASELINE_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "learning_rate": 0.07,
    "n_estimators": 200,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "max_depth": 7,
    "min_child_weight": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

WEIGHTED_SPEC = ValidationSpec(
    name="XGBoost (class-weighted, validated)",
    slug="xgb_weighted_validated",
    balance="weighted",
    estimator_factory=lambda: XGBClassifier(**WEIGHTED_PARAMS),
    hyperparameters=WEIGHTED_PARAMS,
)

BASELINE_SPEC = ValidationSpec(
    name="XGBoost baseline (validated)",
    slug="xgb_baseline_validated",
    balance="downsampled",
    estimator_factory=lambda: XGBClassifier(**BASELINE_PARAMS),
    hyperparameters=BASELINE_PARAMS,
)
