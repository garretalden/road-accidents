"""Load and clean the raw UK accidents CSV; load processed train/test splits."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from . import RANDOM_STATE, RAW_DATA_PATH, TARGET_MAP

DROP_COLUMNS = [
    "Accident_Index",
    "Unnamed: 0",
    "Location_Easting_OSGR",
    "Location_Northing_OSGR",
    "Longitude",
    "Latitude",
    "Local_Authority_(District)",
    "Local_Authority_(Highway)",
    "1st_Road_Number",
    "2nd_Road_Number",
    "Pedestrian_Crossing-Human_Control",
    "Special_Conditions_at_Site",
    "Carriageway_Hazards",
    "LSOA_of_Accident_Location",
    "Year",
    "Junction_Control",
    "Police_Force",
    "Number_of_Vehicles",
    "Number_of_Casualties",
    "Did_Police_Officer_Attend_Scene_of_Accident",
]


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the CSV, drop non-predictive columns, and drop the ~200 rows with any NA."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `make data` or place the "
            "UK_Accident.csv file at that path."
        )
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.dropna()
    return df


def load_split(
    path: Path | str = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Recreate the project's frozen stratified 80/20 train/test split."""
    from .features import add_time_features

    frame = add_time_features(load_raw(path))
    X = frame.drop(columns="Accident_Severity")
    y = frame["Accident_Severity"].map(TARGET_MAP)
    if y.isna().any():
        raise ValueError("Accident_Severity contains values outside 1=Fatal, 2=Serious, 3=Slight")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.to_numpy(dtype=np.int8),
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test
