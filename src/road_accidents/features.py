"""Feature engineering: date/time decomposition and cyclical encoding."""

import numpy as np
import pandas as pd

from .config import RUSH_HOURS, SEASON_MAP


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive month, season, cyclical hour, and rush_hour columns from Date + Time.

    Consumes and drops ``Date`` and ``Time``; keeps ``Month`` (used later for
    season dummies) but drops raw hour after cyclical encoding.
    """
    out = df.copy()
    out["Month"] = pd.to_datetime(out["Date"], format="%d/%m/%Y").dt.month
    out["Season"] = out["Month"].map(SEASON_MAP)

    hour = pd.to_datetime(out["Time"], format="%H:%M").dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["rush_hour"] = hour.isin(RUSH_HOURS).astype(int)

    return out.drop(columns=["Date", "Time"])
