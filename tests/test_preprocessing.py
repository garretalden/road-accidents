"""Tests for the reusable ColumnTransformer preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from road_accidents.preprocessing import (
    NUMERIC_FEATURES,
    ONE_HOT_FEATURES,
    build_preprocessor,
    feature_names,
    transform_to_frame,
)


def _synthetic_frame(n: int = 200, weather_choices=("Fine", "Raining")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Number_of_Vehicles": rng.integers(1, 5, size=n),
            "Speed_limit": rng.choice([20, 30, 40, 60, 70], size=n),
            "Season": rng.choice(["Fall", "Spring", "Summer", "Winter"], size=n),
            "hour_sin": rng.normal(size=n),
            "hour_cos": rng.normal(size=n),
            "rush_hour": rng.integers(0, 2, size=n),
            "Road_Type": rng.choice(["Single carriageway", "Dual carriageway"], size=n),
            "Light_Conditions": rng.choice(["Daylight", "Darkness"], size=n),
            "Weather_Conditions": rng.choice(weather_choices, size=n),
            "Road_Surface_Conditions": rng.choice(["Dry", "Wet/Damp"], size=n),
            "1st_Road_Class": rng.choice([1, 3, 6], size=n),
            "2nd_Road_Class": rng.choice([-1, 3, 6], size=n),
            "Pedestrian_Crossing-Physical_Facilities": rng.choice(
                ["Zebra", "None within 50 metres"], size=n
            ),
            "Day_of_Week": rng.integers(1, 8, size=n),
            "Urban_or_Rural_Area": rng.choice([1, 2], size=n),
        }
    )


def test_categoricals_are_replaced_by_dummies() -> None:
    train = _synthetic_frame()
    preprocessor = build_preprocessor().fit(train)
    encoded = transform_to_frame(preprocessor, train)
    for col in ONE_HOT_FEATURES:
        assert col not in encoded.columns
    assert any(c.startswith("Weather_Conditions_") for c in encoded.columns)


def test_speed_limit_stays_numeric() -> None:
    train = _synthetic_frame()
    preprocessor = build_preprocessor().fit(train)
    encoded = transform_to_frame(preprocessor, train)
    # Passthrough numeric columns keep their original values, not ranks/indices.
    assert list(encoded["Speed_limit"]) == list(train["Speed_limit"].astype(float))
    for col in NUMERIC_FEATURES:
        assert col in encoded.columns


def test_season_dummies_are_mutually_exclusive() -> None:
    train = _synthetic_frame()
    preprocessor = build_preprocessor().fit(train)
    encoded = transform_to_frame(preprocessor, train)
    season_cols = [c for c in encoded.columns if c.startswith("Season_")]
    assert (encoded[season_cols].sum(axis=1) == 1).all()


def test_fit_on_train_transform_on_test_is_consistent() -> None:
    train = _synthetic_frame(n=200)
    test = _synthetic_frame(n=50)
    preprocessor = build_preprocessor().fit(train)

    train_encoded = transform_to_frame(preprocessor, train)
    test_encoded = transform_to_frame(preprocessor, test)

    assert list(train_encoded.columns) == list(test_encoded.columns)
    assert list(test_encoded.columns) == feature_names(preprocessor)


def test_unseen_category_in_test_set_does_not_raise() -> None:
    train = _synthetic_frame(weather_choices=("Fine", "Raining"))
    test = _synthetic_frame(n=10, weather_choices=("Snowing lightly",))
    preprocessor = build_preprocessor().fit(train)

    test_encoded = transform_to_frame(preprocessor, test)

    weather_cols = [c for c in test_encoded.columns if c.startswith("Weather_Conditions_")]
    assert test_encoded[weather_cols].to_numpy().sum() == pytest.approx(0.0)
