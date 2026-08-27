"""Generate model-independent data-quality audits and report-ready EDA outputs."""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from src import FIGURES_DIR, RAW_DATA_PATH, RESULTS_DIR
from src.data import DROP_COLUMNS
from src.features import RUSH_HOURS, SEASON_MAP

SEVERITY_MAP = {1: "Fatal", 2: "Serious", 3: "Slight"}
SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]
SEVERITY_COLORS = {"Fatal": "#D62728", "Serious": "#F2CF5B", "Slight": "#4C78A8"}
DAY_MAP = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
FIRST_ROAD_CLASS_MAP = {
    1: "Motorway",
    2: "A(M)",
    3: "A",
    4: "B",
    5: "C",
    6: "Unclassified",
}
SECOND_ROAD_CLASS_MAP = {-1: "No second road", **FIRST_ROAD_CLASS_MAP}
URBAN_RURAL_MAP = {1: "Urban", 2: "Rural", 3: "Unallocated"}

POST_COLLISION_FIELDS = {
    "Number_of_Vehicles",
    "Number_of_Casualties",
    "Did_Police_Officer_Attend_Scene_of_Accident",
}
IDENTIFIER_FIELDS = {"Accident_Index", "Unnamed: 0"}
GEOGRAPHY_FIELDS = {
    "Location_Easting_OSGR",
    "Location_Northing_OSGR",
    "Longitude",
    "Latitude",
}
HIGH_CARDINALITY_FIELDS = {
    "Local_Authority_(District)",
    "Local_Authority_(Highway)",
    "1st_Road_Number",
    "2nd_Road_Number",
    "LSOA_of_Accident_Location",
    "Police_Force",
}

CATEGORICAL_FEATURES = [
    "Severity",
    "Day of week",
    "Month",
    "Season",
    "Rush hour",
    "Speed limit",
    "First road class",
    "Second road class",
    "Road type",
    "Urban/rural",
    "Crossing facilities",
    "Light conditions",
    "Weather conditions",
    "Road surface",
]


def removal_reason(column: str) -> str:
    """Return the documented category for a field excluded by the project contract."""
    if column in POST_COLLISION_FIELDS:
        return "Post-collision field excluded to prevent target leakage"
    if column in IDENTIFIER_FIELDS:
        return "Row index or collision identifier; not an analytical feature"
    if column in GEOGRAPHY_FIELDS:
        return "Geographic coordinate excluded from the analytical feature contract"
    if column in HIGH_CARDINALITY_FIELDS:
        return "High-cardinality geographic or administrative identifier"
    return "Excluded by the current analytical feature contract; no narrower rationale documented"


def invalid_value_checks(raw: pd.DataFrame) -> pd.DataFrame:
    """Count invalid retained values without changing the raw data."""
    checks = {
        "Severity outside {1, 2, 3}": ~raw["Accident_Severity"].isin(SEVERITY_MAP),
        "Day of week outside 1-7": ~raw["Day_of_Week"].isin(DAY_MAP),
        "First road class outside 1-6": ~raw["1st_Road_Class"].isin(FIRST_ROAD_CLASS_MAP),
        "Second road class outside {-1, 1, ..., 6}": ~raw["2nd_Road_Class"].isin(
            SECOND_ROAD_CLASS_MAP
        ),
        "Non-positive speed limit": raw["Speed_limit"].le(0),
        "Urban/rural code outside {1, 2, 3}": ~raw["Urban_or_Rural_Area"].isin(
            URBAN_RURAL_MAP
        ),
        "Unparseable date": pd.to_datetime(
            raw["Date"], format="%d/%m/%Y", errors="coerce"
        ).isna(),
        "Unparseable non-missing time": (
            pd.to_datetime(raw["Time"], format="%H:%M", errors="coerce").isna()
            & raw["Time"].notna()
        ),
    }
    string_columns = raw.select_dtypes(include=["object", "string"]).columns
    checks["Blank retained string value"] = raw[
        [column for column in string_columns if column not in DROP_COLUMNS]
    ].apply(lambda series: series.str.strip().eq(""), axis=0).any(axis=1)
    return pd.DataFrame(
        {"check": list(checks), "invalid_rows": [int(mask.sum()) for mask in checks.values()]}
    )


def prepare_cleaned_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the project cleaning contract and add readable EDA time/category fields."""
    retained = raw.drop(columns=DROP_COLUMNS, errors="ignore").dropna().copy()
    dates = pd.to_datetime(retained["Date"], format="%d/%m/%Y", errors="raise")
    times = pd.to_datetime(retained["Time"], format="%H:%M", errors="raise")
    hour = times.dt.hour

    cleaned = pd.DataFrame(index=retained.index)
    cleaned["Severity score"] = retained["Accident_Severity"].astype(int)
    cleaned["Severity"] = retained["Accident_Severity"].map(SEVERITY_MAP)
    cleaned["Day of week"] = retained["Day_of_Week"].map(DAY_MAP)
    cleaned["Month number"] = dates.dt.month
    cleaned["Month"] = dates.dt.month.map(lambda value: calendar.month_abbr[value])
    cleaned["Season"] = dates.dt.month.map(SEASON_MAP)
    cleaned["Hour"] = hour
    cleaned["Hour sin"] = np.sin(2 * np.pi * hour / 24)
    cleaned["Hour cos"] = np.cos(2 * np.pi * hour / 24)
    cleaned["Rush hour"] = np.where(hour.isin(RUSH_HOURS), "Yes", "No")
    cleaned["Rush hour flag"] = hour.isin(RUSH_HOURS).astype(int)
    cleaned["Speed limit"] = retained["Speed_limit"].astype(int)
    cleaned["First road class"] = retained["1st_Road_Class"].map(FIRST_ROAD_CLASS_MAP)
    cleaned["Second road class"] = retained["2nd_Road_Class"].map(SECOND_ROAD_CLASS_MAP)
    cleaned["Road type"] = retained["Road_Type"]
    cleaned["Urban/rural"] = retained["Urban_or_Rural_Area"].map(URBAN_RURAL_MAP)
    cleaned["Crossing facilities"] = retained["Pedestrian_Crossing-Physical_Facilities"]
    cleaned["Light conditions"] = retained["Light_Conditions"]
    cleaned["Weather conditions"] = retained["Weather_Conditions"]
    cleaned["Road surface"] = retained["Road_Surface_Conditions"]
    return cleaned


def column_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """Describe every source column and its role in the analytical contract."""
    rows = []
    for column in raw.columns:
        removed = column in DROP_COLUMNS
        rows.append(
            {
                "column": column,
                "dtype": str(raw[column].dtype),
                "missing_count": int(raw[column].isna().sum()),
                "missing_percent": float(raw[column].isna().mean() * 100),
                "unique_nonmissing": int(raw[column].nunique(dropna=True)),
                "status": "Removed" if removed else "Retained",
                "reason": removal_reason(column) if removed else "Retained for cleaning or EDA",
            }
        )
    return pd.DataFrame(rows)


def audit_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return compact raw-to-cleaned reconciliation metrics."""
    retained_columns = [column for column in raw.columns if column not in DROP_COLUMNS]
    metrics = {
        "Raw rows": len(raw),
        "Raw columns": raw.shape[1],
        "Exact duplicate rows": int(raw.duplicated().sum()),
        "Rows with missing values in any raw field": int(raw.isna().any(axis=1).sum()),
        "Rows with missing values in retained fields": int(
            raw[retained_columns].isna().any(axis=1).sum()
        ),
        "Rows removed by complete-case cleaning": len(raw) - len(cleaned),
        "Cleaned rows": len(cleaned),
        "Retained source columns before feature derivation": len(retained_columns),
        "Derived analytical columns": cleaned.shape[1],
    }
    if "Accident_Index" in raw:
        metrics["Rows with a repeated Accident_Index"] = int(raw["Accident_Index"].duplicated().sum())
    return pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})


def severity_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    counts = cleaned["Severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    return pd.DataFrame(
        {
            "severity": counts.index,
            "count": counts.to_numpy(),
            "percent": counts.to_numpy() / len(cleaned) * 100,
        }
    )


def category_severity_summary(
    cleaned: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """Return counts and within-level severity shares for categorical features."""
    rows = []
    for feature in features or CATEGORICAL_FEATURES[1:]:
        table = pd.crosstab(cleaned[feature], cleaned["Severity"]).reindex(
            columns=SEVERITY_ORDER, fill_value=0
        )
        for level, values in table.iterrows():
            total = int(values.sum())
            row = {
                "feature": feature,
                "level": level,
                "count": total,
                "percent_of_cleaned_rows": total / len(cleaned) * 100,
            }
            for severity in SEVERITY_ORDER:
                row[f"{severity.lower()}_count"] = int(values[severity])
                row[f"{severity.lower()}_share_percent"] = (
                    float(values[severity] / total * 100) if total else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Compute bias-corrected Cramer's V for two categorical series."""
    table = pd.crosstab(left, right)
    n = int(table.to_numpy().sum())
    if n == 0 or min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    rows, columns = table.shape
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(columns_corrected - 1, rows_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else 0.0


def categorical_associations(cleaned: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(
        np.eye(len(CATEGORICAL_FEATURES)),
        index=CATEGORICAL_FEATURES,
        columns=CATEGORICAL_FEATURES,
    )
    for left_index, left in enumerate(CATEGORICAL_FEATURES):
        for right in CATEGORICAL_FEATURES[left_index + 1 :]:
            value = cramers_v(cleaned[left], cleaned[right])
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
    return matrix


def numeric_correlations(cleaned: pd.DataFrame) -> pd.DataFrame:
    columns = ["Severity score", "Speed limit", "Hour sin", "Hour cos", "Rush hour flag"]
    return cleaned[columns].corr(method="spearman")


def _save_figure(
    fig: plt.Figure, path: Path, *, rect: tuple[float, float, float, float] | None = None
) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_severity_distribution(cleaned: pd.DataFrame, path: Path) -> None:
    summary = severity_summary(cleaned)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        summary["severity"],
        summary["count"],
        color=[SEVERITY_COLORS[value] for value in summary["severity"]],
    )
    for bar, row in zip(bars, summary.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row.count:,}\n({row.percent:.2f}%)",
            ha="center",
            va="bottom",
        )
    ax.set(title="Recorded accidents by severity", xlabel="Severity", ylabel="Accidents")
    ax.ticklabel_format(axis="y", style="plain")
    _save_figure(fig, path)


def _count_bars(ax: plt.Axes, series: pd.Series, order: list, title: str) -> None:
    counts = series.value_counts().reindex(order, fill_value=0)
    ax.bar([str(value) for value in counts.index], counts.to_numpy(), color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel("Recorded accidents")
    ax.tick_params(axis="x", rotation=45)
    ax.ticklabel_format(axis="y", style="plain")


def plot_temporal_distributions(cleaned: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    _count_bars(axes[0, 0], cleaned["Month"], list(calendar.month_abbr)[1:], "By month")
    _count_bars(axes[0, 1], cleaned["Season"], SEASON_ORDER, "By season")
    _count_bars(axes[1, 0], cleaned["Day of week"], DAY_ORDER, "By weekday")
    _count_bars(axes[1, 1], cleaned["Hour"], list(range(24)), "By hour of day")
    fig.suptitle("Temporal distribution of recorded accidents", fontsize=16)
    _save_figure(fig, path)


def plot_weekday_hour_heatmap(cleaned: pd.DataFrame, path: Path) -> None:
    table = pd.crosstab(cleaned["Day of week"], cleaned["Hour"]).reindex(
        index=DAY_ORDER, columns=range(24), fill_value=0
    )
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(table / 1000, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Accidents (thousands)"})
    ax.set(title="Recorded accidents by weekday and hour", xlabel="Hour of day", ylabel="")
    _save_figure(fig, path)


def _severity_composition(
    ax: plt.Axes, cleaned: pd.DataFrame, feature: str, order: list, title: str
) -> None:
    counts = pd.crosstab(cleaned[feature], cleaned["Severity"]).reindex(
        index=order, columns=SEVERITY_ORDER, fill_value=0
    )
    proportions = counts.div(counts.sum(axis=1), axis=0) * 100
    left = np.zeros(len(proportions))
    for severity in SEVERITY_ORDER:
        ax.barh(
            range(len(proportions)),
            proportions[severity],
            left=left,
            color=SEVERITY_COLORS[severity],
            label=severity,
        )
        left += proportions[severity].to_numpy()
    labels = [f"{value} (n={int(total):,})" for value, total in zip(order, counts.sum(axis=1))]
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.invert_yaxis()
    ax.set(title=title, xlabel="Within-group severity share (%)", xlim=(0, 100))


def _figure_legend(fig: plt.Figure) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEVERITY_COLORS[name]) for name in SEVERITY_ORDER]
    fig.legend(
        handles,
        SEVERITY_ORDER,
        title="Severity",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
    )


def plot_road_context(cleaned: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    _severity_composition(
        axes[0], cleaned, "Speed limit", sorted(cleaned["Speed limit"].unique()), "Speed limit"
    )
    _severity_composition(
        axes[1],
        cleaned,
        "Road type",
        cleaned["Road type"].value_counts().index.tolist(),
        "Road type",
    )
    _severity_composition(
        axes[2], cleaned, "Urban/rural", ["Urban", "Rural", "Unallocated"], "Urban/rural"
    )
    fig.suptitle("Road context and severity among recorded accidents", fontsize=16, y=0.995)
    _figure_legend(fig)
    _save_figure(fig, path, rect=(0, 0, 1, 0.90))


def plot_road_classes(cleaned: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    _severity_composition(
        axes[0], cleaned, "First road class", list(FIRST_ROAD_CLASS_MAP.values()), "First road class"
    )
    _severity_composition(
        axes[1],
        cleaned,
        "Second road class",
        list(SECOND_ROAD_CLASS_MAP.values()),
        "Second road class",
    )
    fig.suptitle("Road class and severity among recorded accidents", fontsize=16, y=0.995)
    _figure_legend(fig)
    _save_figure(fig, path, rect=(0, 0, 1, 0.90))


def plot_environment(cleaned: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    for ax, feature, title in zip(
        axes,
        ["Light conditions", "Weather conditions", "Road surface"],
        ["Light conditions", "Weather conditions", "Road surface conditions"],
    ):
        order = cleaned[feature].value_counts().index.tolist()
        _severity_composition(ax, cleaned, feature, order, title)
    fig.suptitle(
        "Environmental conditions and severity among recorded accidents", fontsize=16, y=0.995
    )
    _figure_legend(fig)
    _save_figure(fig, path, rect=(0, 0, 1, 0.90))


def plot_crossing_facilities(cleaned: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    order = cleaned["Crossing facilities"].value_counts().index.tolist()
    _severity_composition(ax, cleaned, "Crossing facilities", order, "Pedestrian-crossing facilities")
    ax.legend(title="Severity", loc="lower right")
    _save_figure(fig, path)


def plot_matrix(
    matrix: pd.DataFrame, path: Path, *, title: str, figsize: tuple[int, int], annotate: bool
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        vmin=-1 if matrix.to_numpy().min() < 0 else 0,
        vmax=1,
        center=0 if matrix.to_numpy().min() < 0 else None,
        cmap="coolwarm" if matrix.to_numpy().min() < 0 else "Blues",
        annot=annotate,
        fmt=".2f",
        square=True,
        ax=ax,
    )
    ax.set_title(title)
    _save_figure(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Path to UK_Accident.csv")
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Raw data not found at {args.data}; run `make data` first")

    print(f"[load] reading {args.data}")
    raw = pd.read_csv(args.data, low_memory=False)
    cleaned = prepare_cleaned_data(raw)
    figure_dir = FIGURES_DIR / "eda"
    result_dir = RESULTS_DIR / "eda"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("[tables] generating quality audits and descriptive summaries")
    audit_summary(raw, cleaned).to_csv(result_dir / "audit_summary.csv", index=False)
    column_audit(raw).to_csv(result_dir / "column_audit.csv", index=False)
    invalid_value_checks(raw).to_csv(result_dir / "invalid_value_checks.csv", index=False)
    severity_summary(cleaned).to_csv(result_dir / "severity_summary.csv", index=False)
    cleaned[["Severity score", "Speed limit", "Month number", "Hour", "Rush hour flag"]].describe().T.to_csv(
        result_dir / "numeric_summary.csv"
    )
    category_severity_summary(cleaned).to_csv(
        result_dir / "category_by_severity.csv", index=False
    )
    correlations = numeric_correlations(cleaned)
    correlations.to_csv(result_dir / "spearman_correlations.csv")
    associations = categorical_associations(cleaned)
    associations.to_csv(result_dir / "categorical_associations.csv")

    print("[figures] generating nine report-ready EDA figures")
    sns.set_theme(style="whitegrid")
    plot_severity_distribution(cleaned, figure_dir / "severity_distribution.png")
    plot_temporal_distributions(cleaned, figure_dir / "temporal_distributions.png")
    plot_weekday_hour_heatmap(cleaned, figure_dir / "weekday_hour_heatmap.png")
    plot_road_context(cleaned, figure_dir / "road_context_by_severity.png")
    plot_road_classes(cleaned, figure_dir / "road_class_by_severity.png")
    plot_environment(cleaned, figure_dir / "environment_by_severity.png")
    plot_crossing_facilities(cleaned, figure_dir / "crossing_facilities_by_severity.png")
    plot_matrix(
        correlations,
        figure_dir / "spearman_correlation.png",
        title="Spearman correlations among ordered and numeric fields",
        figsize=(8, 7),
        annotate=True,
    )
    plot_matrix(
        associations,
        figure_dir / "categorical_associations.png",
        title="Categorical associations (bias-corrected Cramér's V)",
        figsize=(15, 13),
        annotate=False,
    )
    print(f"[done] EDA figures: {figure_dir.relative_to(Path.cwd())}")
    print(f"[done] EDA tables: {result_dir.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
