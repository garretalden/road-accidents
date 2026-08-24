import pandas as pd
import pytest

from src.preprocessing import MODEL_FEATURES, build_preprocessor, feature_names, validate_pre_accident_columns


def sample_frame():
    return pd.DataFrame([{
        "Road_Type": 1, "Light_Conditions": 1, "Weather_Conditions": 1,
        "Road_Surface_Conditions": 1, "1st_Road_Class": 3, "2nd_Road_Class": -1,
        "Pedestrian_Crossing-Physical_Facilities": 0, "Day_of_Week": 2,
        "Urban_or_Rural_Area": 1, "Season": "Winter", "Speed_limit": 30,
        "hour_sin": 0.5, "hour_cos": -0.5, "rush_hour": 1,
    }])


def test_preprocessor_contract_excludes_post_collision_fields():
    assert "Number_of_Vehicles" not in MODEL_FEATURES
    assert "Number_of_Casualties" not in MODEL_FEATURES
    validate_pre_accident_columns(sample_frame())


def test_preprocessor_rejects_post_collision_fields():
    frame = sample_frame().assign(Number_of_Vehicles=2)
    with pytest.raises(ValueError, match="post-collision"):
        validate_pre_accident_columns(frame)


def test_preprocessor_handles_unknown_categories_and_names_outputs():
    frame = sample_frame()
    transformer = build_preprocessor().fit(frame)
    unknown = frame.copy()
    unknown["Weather_Conditions"] = 999
    transformed = transformer.transform(unknown)
    assert transformed.shape[1] == len(feature_names(transformer))
