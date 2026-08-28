"""Fail-fast validation for the audited final-training dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import RANDOM_STATE
from .data import DROP_COLUMNS, deduplicate_raw_records


@dataclass(frozen=True)
class DatasetContract:
    size_bytes: int
    sha256: str
    columns: tuple[str, ...]
    raw_rows: int
    duplicate_rows: int
    cleaned_rows: int
    years: tuple[int, ...]
    class_counts: dict[int, int]
    train_rows: int
    test_rows: int
    test_class_counts: dict[int, int]


FINAL_DATASET_CONTRACT = DatasetContract(
    size_bytes=449_667_447,
    sha256="a387b49d22a06191bcec5bc0c46c29094a62cd2c57a361b69dbdce94cc922799",
    columns=(
        "Unnamed: 0",
        "Accident_Index",
        "Location_Easting_OSGR",
        "Location_Northing_OSGR",
        "Longitude",
        "Latitude",
        "Police_Force",
        "Accident_Severity",
        "Number_of_Vehicles",
        "Number_of_Casualties",
        "Date",
        "Day_of_Week",
        "Time",
        "Local_Authority_(District)",
        "Local_Authority_(Highway)",
        "1st_Road_Class",
        "1st_Road_Number",
        "Road_Type",
        "Speed_limit",
        "Junction_Control",
        "2nd_Road_Class",
        "2nd_Road_Number",
        "Pedestrian_Crossing-Human_Control",
        "Pedestrian_Crossing-Physical_Facilities",
        "Light_Conditions",
        "Weather_Conditions",
        "Road_Surface_Conditions",
        "Special_Conditions_at_Site",
        "Carriageway_Hazards",
        "Urban_or_Rural_Area",
        "Did_Police_Officer_Attend_Scene_of_Accident",
        "LSOA_of_Accident_Location",
        "Year",
    ),
    raw_rows=1_504_150,
    duplicate_rows=34_155,
    cleaned_rows=1_469_845,
    years=(2005, 2006, 2007, 2009, 2010, 2011, 2012, 2013, 2014),
    class_counts={1: 19_039, 2: 198_894, 3: 1_251_912},
    train_rows=1_175_876,
    test_rows=293_969,
    test_class_counts={1: 3_808, 2: 39_779, 3: 250_382},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def validate_dataset(path: Path, contract: DatasetContract = FINAL_DATASET_CONTRACT) -> dict:
    """Validate provenance, cleaning reconciliation, and deterministic split counts."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    _require_equal("file size", path.stat().st_size, contract.size_bytes)
    _require_equal("SHA-256", sha256_file(path), contract.sha256)

    raw = pd.read_csv(path, low_memory=False)
    _require_equal("source columns", tuple(raw.columns), contract.columns)
    _require_equal("raw rows", len(raw), contract.raw_rows)

    deduplicated = deduplicate_raw_records(raw)
    _require_equal(
        "duplicate substantive rows",
        len(raw) - len(deduplicated),
        contract.duplicate_rows,
    )
    cleaned = deduplicated.drop(columns=DROP_COLUMNS, errors="ignore").dropna()
    _require_equal("cleaned rows", len(cleaned), contract.cleaned_rows)
    _require_equal(
        "represented years",
        tuple(sorted(raw["Year"].astype(int).unique())),
        contract.years,
    )
    class_counts = {
        int(label): int(count)
        for label, count in cleaned["Accident_Severity"].value_counts().sort_index().items()
    }
    _require_equal("severity class counts", class_counts, contract.class_counts)

    train_y, test_y = train_test_split(
        cleaned["Accident_Severity"],
        test_size=0.2,
        stratify=cleaned["Accident_Severity"],
        random_state=RANDOM_STATE,
    )
    _require_equal("training rows", len(train_y), contract.train_rows)
    _require_equal("test rows", len(test_y), contract.test_rows)
    test_counts = {
        int(label): int(count)
        for label, count in test_y.value_counts().sort_index().items()
    }
    _require_equal("test severity class counts", test_counts, contract.test_class_counts)
    return {
        "sha256": contract.sha256,
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(deduplicated),
        "cleaned_rows": len(cleaned),
        "train_rows": len(train_y),
        "test_rows": len(test_y),
    }
