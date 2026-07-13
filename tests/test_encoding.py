"""Tests for the categorical / ordinal encoding pipeline."""

import numpy as np
import pandas as pd
import pytest

from road_accidents.config import CATEGORICAL_FEATURES
from road_accidents.encoding import fit_transform, transform_row


def _synthetic_frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "Accident_Severity": rng.choice([1, 2, 3], size=n),
            "Number_of_Vehicles": rng.integers(1, 5, size=n),
            "Speed_limit": rng.choice([20, 30, 40, 60, 70], size=n),
            "Month": rng.integers(1, 13, size=n),
            "Season": rng.choice(["Fall", "Spring", "Summer", "Winter"], size=n),
            "hour_sin": rng.normal(size=n),
            "hour_cos": rng.normal(size=n),
            "rush_hour": rng.integers(0, 2, size=n),
            "Road_Type": rng.choice(["Single carriageway", "Dual carriageway"], size=n),
            "Light_Conditions": rng.choice(["Daylight", "Darkness"], size=n),
            "Weather_Conditions": rng.choice(["Fine", "Raining"], size=n),
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
    return frame


def test_categoricals_are_replaced_by_dummies() -> None:
    df = _synthetic_frame()
    encoded, encoders = fit_transform(df)
    for col in CATEGORICAL_FEATURES:
        assert col not in encoded.columns
    # Every OHE-generated column should exist in the encoded frame.
    for col in encoders.ohe.get_feature_names_out(CATEGORICAL_FEATURES):
        assert col in encoded.columns


def test_speed_limit_is_ordinal_ordered() -> None:
    df = _synthetic_frame()
    encoded, encoders = fit_transform(df)
    categories = list(encoders.ordinal.categories_[0])
    assert categories == sorted(categories)
    assert set(encoded["Speed_limit"]).issubset(set(range(len(categories))))


def test_season_dummies_are_mutually_exclusive() -> None:
    df = _synthetic_frame()
    encoded, _ = fit_transform(df)
    season_cols = ["Fall", "Spring", "Summer", "Winter"]
    assert (encoded[season_cols].sum(axis=1) == 1).all()


def test_transform_row_matches_training_columns() -> None:
    df = _synthetic_frame()
    _, encoders = fit_transform(df)
    example = df.drop(columns="Accident_Severity").iloc[0].to_dict()
    row = transform_row(example, encoders)
    assert list(row.columns) == encoders.feature_columns
    assert row.shape == (1, len(encoders.feature_columns))


def test_transform_row_handles_unseen_category() -> None:
    df = _synthetic_frame()
    _, encoders = fit_transform(df)
    example = df.drop(columns="Accident_Severity").iloc[0].to_dict()
    example["Weather_Conditions"] = "Snowing lightly"  # not in training vocab
    row = transform_row(example, encoders)
    # handle_unknown='ignore' → all zeros for that group, but no crash.
    weather_cols = [c for c in row.columns if c.startswith("Weather_Conditions_")]
    assert row[weather_cols].sum().sum() == pytest.approx(0.0)
