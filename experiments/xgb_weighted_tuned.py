"""Factory for the selected full-data, class-weighted XGBoost experiment."""

from xgboost import XGBClassifier

from road_accidents.tuning import BASE_PARAMS
from road_accidents.validation import ValidationSpec


def make_tuned_spec(parameters: dict) -> ValidationSpec:
    params = {**BASE_PARAMS, **parameters}
    return ValidationSpec(
        name="XGBoost (class-weighted, tuned)",
        slug="xgb_weighted_tuned",
        balance="weighted",
        estimator_factory=lambda: XGBClassifier(**params),
        hyperparameters=params,
    )
