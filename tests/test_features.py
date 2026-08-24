import numpy as np
import pandas as pd

from src.features import add_time_features, hour_to_cyclical, is_rush_hour, month_to_season


def test_time_feature_helpers():
    assert month_to_season(1) == "Winter"
    assert month_to_season(7) == "Summer"
    sine, cosine = hour_to_cyclical(0)
    assert sine == 0
    assert cosine == 1
    assert is_rush_hour(8) == 1
    assert is_rush_hour(12) == 0


def test_add_time_features_removes_raw_timestamp_fields():
    frame = pd.DataFrame({"Date": ["01/01/2018"], "Time": ["17:30"], "value": [1]})
    result = add_time_features(frame)
    assert "Date" not in result and "Time" not in result
    assert result.loc[0, "Season"] == "Winter"
    assert result.loc[0, "rush_hour"] == 1
    assert np.isfinite(result.loc[0, ["hour_sin", "hour_cos"]].astype(float)).all()
