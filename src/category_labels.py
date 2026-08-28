"""Human-readable labels for coded categorical values."""

from __future__ import annotations

from typing import Any


CATEGORY_LABELS = {
    "Road_Type": {
        1: "Roundabout", 2: "One-way street", 3: "Dual carriageway",
        6: "Single carriageway", 7: "Slip road", 9: "Unknown",
    },
    "Light_Conditions": {
        1: "Daylight", 4: "Dark: lights lit", 5: "Dark: lights unlit",
        6: "Dark: no lighting", 7: "Dark: unknown lighting",
    },
    "Weather_Conditions": {
        1: "Fine", 2: "Rain", 3: "Snow", 4: "Fine + high winds",
        5: "Rain + high winds", 6: "Snow + high winds", 7: "Fog/mist",
        8: "Other", 9: "Unknown",
    },
    "Road_Surface_Conditions": {
        1: "Dry", 2: "Wet/damp", 3: "Snow", 4: "Frost/ice",
        5: "Flood", 9: "Unknown",
    },
    "1st_Road_Class": {
        1: "Motorway", 2: "A(M)", 3: "A", 4: "B", 5: "C", 6: "Unclassified",
    },
    "2nd_Road_Class": {
        -1: "No second road", 1: "Motorway", 2: "A(M)", 3: "A",
        4: "B", 5: "C", 6: "Unclassified",
    },
    "Day_of_Week": {
        1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
        5: "Thursday", 6: "Friday", 7: "Saturday",
    },
    "Urban_or_Rural_Area": {1: "Urban", 2: "Rural", 3: "Unallocated"},
    "rush_hour": {0: "No", 1: "Yes"},
}


def format_category_option(feature: str, value: Any) -> str:
    """Format a coded option while preserving its original value for selection."""
    label = CATEGORY_LABELS.get(feature, {}).get(value)
    return f"{value} — {label}" if label is not None else str(value)
