"""Feature engineering: date/time decomposition and cyclical encoding.

The helpers below are written to work identically on a scalar hour/month (for
single-row inference, e.g. the Streamlit app) or a vectorized ``pd.Series``
(for bulk training data), so there is exactly one implementation of each
derived feature.
"""

import numpy as np
import pandas as pd

SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer",
    7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall", 12: "Winter",
}
RUSH_HOURS = frozenset({7, 8, 9, 16, 17, 18})


def month_to_season(month):
    """Map a month (1-12) or Series of months to its season name."""
    if isinstance(month, pd.Series):
        return month.map(SEASON_MAP)
    return SEASON_MAP[month]


def hour_to_cyclical(hour):
    """Map an hour-of-day (0-23) or Series of hours to (sin, cos) components."""
    sin = np.sin(2 * np.pi * hour / 24)
    cos = np.cos(2 * np.pi * hour / 24)
    return sin, cos


def is_rush_hour(hour):
    """Return 1/0 (or a 0/1 Series) for whether hour falls in RUSH_HOURS."""
    if isinstance(hour, pd.Series):
        return hour.isin(RUSH_HOURS).astype(int)
    return int(hour in RUSH_HOURS)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive month, season, cyclical hour, and rush_hour columns from Date + Time.

    Consumes and drops ``Date`` and ``Time``; keeps ``Month`` (used later for
    season dummies) but drops raw hour after cyclical encoding.
    """
    out = df.copy()
    out["Month"] = pd.to_datetime(out["Date"], format="%d/%m/%Y").dt.month
    out["Season"] = month_to_season(out["Month"])

    hour = pd.to_datetime(out["Time"], format="%H:%M").dt.hour
    out["hour_sin"], out["hour_cos"] = hour_to_cyclical(hour)
    out["rush_hour"] = is_rush_hour(hour)

    return out.drop(columns=["Date", "Time"])
