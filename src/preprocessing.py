"""Reusable scikit-learn preprocessing for the accident-severity models.

A single ``ColumnTransformer`` one-hot encodes the true categorical columns
(plus the derived ``Season``) and passes numeric columns (including
``Speed_limit``, which stays numeric rather than being ordinal-encoded)
through unchanged. Every model receives this preprocessor inside its fitted
pipeline so category discovery remains local to each training fold.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = [
    "Road_Type",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "1st_Road_Class",
    "2nd_Road_Class",
    "Pedestrian_Crossing-Physical_Facilities",
    "Day_of_Week",
    "Urban_or_Rural_Area",
]
NUMERIC_FEATURES = ["Speed_limit", "hour_sin", "hour_cos", "rush_hour"]
ONE_HOT_FEATURES = [*CATEGORICAL_FEATURES, "Season"]
MODEL_FEATURES = [*ONE_HOT_FEATURES, *NUMERIC_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Return an unfit ColumnTransformer for the raw, time-featured columns."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONE_HOT_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def validate_pre_accident_columns(X: pd.DataFrame) -> None:
    """Reject known post-collision variables before any estimator is fitted."""
    forbidden = {
        "Number_of_Vehicles",
        "Number_of_Casualties",
        "Did_Police_Officer_Attend_Scene_of_Accident",
        "Accident_Severity",
    }
    present = sorted(forbidden.intersection(X.columns))
    if present:
        raise ValueError(f"post-collision columns are not valid model inputs: {present}")
    missing = sorted(set(MODEL_FEATURES).difference(X.columns))
    if missing:
        raise ValueError(f"required prediction-time columns are missing: {missing}")


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Clean output column names (strip the cat__/num__ prefixes sklearn adds)."""
    return [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]


def transform_to_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform X with a fitted preprocessor and return a named, indexed DataFrame."""
    transformed = preprocessor.transform(X)
    return pd.DataFrame(transformed, columns=feature_names(preprocessor), index=X.index)
