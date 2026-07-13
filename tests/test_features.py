"""Tests for date/time feature engineering."""

import numpy as np
import pandas as pd
import pytest

from road_accidents.features import add_time_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["01/01/2015", "15/07/2016", "31/12/2017"],
            "Time": ["00:00", "12:00", "17:30"],
            "Number_of_Vehicles": [1, 2, 3],
        }
    )


def test_drops_date_and_time(sample_df: pd.DataFrame) -> None:
    out = add_time_features(sample_df)
    assert "Date" not in out.columns
    assert "Time" not in out.columns


def test_month_extraction(sample_df: pd.DataFrame) -> None:
    out = add_time_features(sample_df)
    assert out["Month"].tolist() == [1, 7, 12]


def test_season_mapping(sample_df: pd.DataFrame) -> None:
    out = add_time_features(sample_df)
    assert out["Season"].tolist() == ["Winter", "Summer", "Winter"]


def test_cyclical_hour_boundaries(sample_df: pd.DataFrame) -> None:
    """Midnight should have sin≈0 and cos≈1; noon should have sin≈0 and cos≈-1."""
    out = add_time_features(sample_df)
    assert out["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert out["hour_sin"].iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert out["hour_cos"].iloc[1] == pytest.approx(-1.0, abs=1e-9)


def test_rush_hour_flag(sample_df: pd.DataFrame) -> None:
    out = add_time_features(sample_df)
    # 00:00 → no, 12:00 → no, 17:30 → yes
    assert out["rush_hour"].tolist() == [0, 0, 1]


def test_hour_unit_circle(sample_df: pd.DataFrame) -> None:
    out = add_time_features(sample_df)
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)
