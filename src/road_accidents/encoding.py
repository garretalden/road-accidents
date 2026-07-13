"""One-hot / ordinal encoding of feature columns.

Encoders are exposed so the Streamlit app can transform new single-row inputs
identically to what the training pipeline produced.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .config import CATEGORICAL_FEATURES


@dataclass
class FittedEncoders:
    """Bundle of encoders + column order needed to transform new rows."""

    ohe: OneHotEncoder
    ordinal: OrdinalEncoder
    season_categories: list[str]
    feature_columns: list[str]


def fit_transform(df: pd.DataFrame) -> tuple[pd.DataFrame, FittedEncoders]:
    """One-hot encode categoricals, ordinal-encode Speed_limit, dummy-encode Season.

    Returns the fully encoded frame (still including ``Accident_Severity``) and
    the fitted encoders so the same transformation can be applied to unseen
    input at inference time.
    """
    working = df.copy()

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_arr = ohe.fit_transform(working[CATEGORICAL_FEATURES])
    ohe_df = pd.DataFrame(
        ohe_arr,
        columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES),
        index=working.index,
    )
    working = working.drop(columns=CATEGORICAL_FEATURES)
    working = pd.concat([working, ohe_df], axis=1)

    ordinal = OrdinalEncoder(categories=[sorted(working["Speed_limit"].unique())])
    working["Speed_limit"] = ordinal.fit_transform(working[["Speed_limit"]])

    season_categories = ["Fall", "Spring", "Summer", "Winter"]
    for season in season_categories:
        working[season] = (working["Season"] == season).astype(int)
    working = working.drop(columns=["Season", "Month"])

    feature_columns = [c for c in working.columns if c != "Accident_Severity"]
    encoders = FittedEncoders(
        ohe=ohe,
        ordinal=ordinal,
        season_categories=season_categories,
        feature_columns=feature_columns,
    )
    return working, encoders


def transform_row(row: dict, encoders: FittedEncoders) -> pd.DataFrame:
    """Encode a single raw input row (as a dict of column → value) for inference.

    The input dict must include all columns the training pipeline expects
    *after* time-feature engineering: the CATEGORICAL_FEATURES plus
    ``Number_of_Vehicles``, ``Speed_limit``, ``Month``, ``Season``,
    ``hour_sin``, ``hour_cos``, ``rush_hour``.
    """
    df = pd.DataFrame([row])

    ohe_arr = encoders.ohe.transform(df[CATEGORICAL_FEATURES])
    ohe_df = pd.DataFrame(
        ohe_arr,
        columns=encoders.ohe.get_feature_names_out(CATEGORICAL_FEATURES),
        index=df.index,
    )
    df = df.drop(columns=CATEGORICAL_FEATURES)
    df = pd.concat([df, ohe_df], axis=1)

    df["Speed_limit"] = encoders.ordinal.transform(df[["Speed_limit"]])

    for season in encoders.season_categories:
        df[season] = (df["Season"] == season).astype(int)
    df = df.drop(columns=["Season", "Month"])

    # Reindex to the exact training column order; any missing dummy columns become 0.
    return df.reindex(columns=encoders.feature_columns, fill_value=0).astype(np.float64)
