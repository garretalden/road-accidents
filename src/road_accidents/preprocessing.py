"""Reusable scikit-learn preprocessing for the accident-severity models.

A single ``ColumnTransformer`` one-hot encodes the true categorical columns
(plus the derived ``Season``) and passes numeric columns (including
``Speed_limit``, which stays numeric rather than being ordinal-encoded)
through unchanged. It must be fit on training data only — see
``scripts/prepare_data.py`` — and can then be reused as-is by any baseline or
experiment model; model-specific steps (e.g. scaling for Logistic Regression)
are added downstream in that model's own ``Pipeline``.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from .config import CATEGORICAL_FEATURES

NUMERIC_FEATURES = ["Speed_limit", "Number_of_Vehicles", "hour_sin", "hour_cos", "rush_hour"]
ONE_HOT_FEATURES = [*CATEGORICAL_FEATURES, "Season"]


def build_preprocessor() -> ColumnTransformer:
    """Return an unfit ColumnTransformer for the raw, time-featured columns."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONE_HOT_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Clean output column names (strip the cat__/num__ prefixes sklearn adds)."""
    return [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]


def transform_to_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform X with a fitted preprocessor and return a named, indexed DataFrame."""
    transformed = preprocessor.transform(X)
    return pd.DataFrame(transformed, columns=feature_names(preprocessor), index=X.index)
