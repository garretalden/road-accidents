"""Load and clean the raw UK accidents CSV; load processed train/test splits."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config import DROP_COLUMNS, PROCESSED_DIR, RAW_CSV_PATH


def load_raw(path: Path | str = RAW_CSV_PATH) -> pd.DataFrame:
    """Read the CSV, drop non-predictive columns, and drop the ~200 rows with any NA."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `make data` or place the "
            "UK_Accident.csv file at that path."
        )
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLUMNS)
    df = df.dropna()
    return df


def load_processed(
    balance: Literal["downsampled", "full"] = "downsampled",
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the preprocessed train/test split produced by ``scripts/prepare_data.py``.

    ``balance="downsampled"`` (the baseline default) loads the train set with
    Slight/Serious undersampled per ``DOWNSAMPLE_TARGETS``. ``balance="full"``
    loads the train set at its natural class distribution, for experiments
    that handle imbalance via class/sample weighting instead. The test set is
    the same either way — it is never resampled or weighted.
    """
    train_stem = "X_train" if balance == "downsampled" else "X_train_full"
    label_stem = "y_train" if balance == "downsampled" else "y_train_full"

    X_train = pd.read_parquet(processed_dir / f"{train_stem}.parquet")
    X_test = pd.read_parquet(processed_dir / "X_test.parquet")
    y_train = pd.read_parquet(processed_dir / f"{label_stem}.parquet")["y"].to_numpy()
    y_test = pd.read_parquet(processed_dir / "y_test.parquet")["y"].to_numpy()
    return X_train, X_test, y_train, y_test
