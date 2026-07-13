"""Load and clean the raw UK accidents CSV."""

from pathlib import Path

import pandas as pd

from .config import DROP_COLUMNS, RAW_CSV_PATH


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
