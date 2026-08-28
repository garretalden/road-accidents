import numpy as np
import pandas as pd
import pytest

from scripts.generate_eda import (
    audit_summary,
    category_severity_summary,
    cramers_v,
    invalid_value_checks,
    prepare_cleaned_data,
)


def sample_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Accident_Index": ["a", "b", "c", "d"],
            "Accident_Severity": [1, 2, 3, 3],
            "Date": ["01/01/2014", "02/02/2014", "03/03/2014", "04/04/2014"],
            "Time": ["08:00", "12:00", "17:00", None],
            "Day_of_Week": [4, 5, 6, 7],
            "1st_Road_Class": [1, 3, 6, 6],
            "Road_Type": ["Roundabout", "Single carriageway", "Slip road", "Slip road"],
            "Speed_limit": [30, 40, 60, 30],
            "2nd_Road_Class": [-1, 3, 6, -1],
            "Pedestrian_Crossing-Physical_Facilities": ["None", "Zebra", "None", "None"],
            "Light_Conditions": ["Daylight", "Daylight", "Dark", "Dark"],
            "Weather_Conditions": ["Fine", "Rain", "Fine", "Fine"],
            "Road_Surface_Conditions": ["Dry", "Wet", "Dry", "Dry"],
            "Urban_or_Rural_Area": [1, 1, 2, 2],
        }
    )


def test_cleaning_and_audit_reconcile_removed_rows():
    raw = sample_raw()
    cleaned = prepare_cleaned_data(raw)
    assert len(cleaned) == 3
    summary = audit_summary(raw, cleaned).set_index("metric")["value"]
    assert summary["Raw rows"] == 4
    assert summary["Duplicate substantive rows removed"] == 0
    assert summary["Rows after substantive deduplication"] == 4
    assert summary["Rows removed by complete-case cleaning"] == 1
    assert summary["Cleaned rows"] == 3


def test_invalid_value_checks_detect_domain_and_parse_failures():
    raw = sample_raw()
    raw.loc[0, "Accident_Severity"] = 9
    raw.loc[1, "Day_of_Week"] = 8
    raw.loc[2, "Time"] = "25:99"
    checks = invalid_value_checks(raw).set_index("check")["invalid_rows"]
    assert checks["Severity outside {1, 2, 3}"] == 1
    assert checks["Day of week outside 1-7"] == 1
    assert checks["Unparseable non-missing time"] == 1


def test_category_severity_shares_sum_to_100_percent():
    cleaned = prepare_cleaned_data(sample_raw())
    summary = category_severity_summary(cleaned, ["Road type"])
    shares = summary[["fatal_share_percent", "serious_share_percent", "slight_share_percent"]]
    assert np.allclose(shares.sum(axis=1), 100)
    assert summary["count"].sum() == len(cleaned)


def test_cramers_v_is_symmetric_bounded_and_detects_identity():
    left = pd.Series(["a", "a", "b", "b", "c", "c"])
    right = pd.Series([1, 1, 2, 2, 3, 3])
    value = cramers_v(left, right)
    assert value == pytest.approx(1.0)
    assert value == pytest.approx(cramers_v(right, left))
    assert 0 <= value <= 1


def test_cleaning_deduplicates_before_dropping_source_columns():
    raw = sample_raw()
    raw["Unnamed: 0"] = [10, 11, 12, 13]
    duplicate = raw.iloc[[0]].copy()
    duplicate["Unnamed: 0"] = 999
    combined = pd.concat([raw, duplicate], ignore_index=True)

    cleaned = prepare_cleaned_data(combined)
    summary = audit_summary(combined, cleaned).set_index("metric")["value"]

    assert len(cleaned) == 3
    assert summary["Duplicate substantive rows removed"] == 1
    assert summary["Rows after substantive deduplication"] == 4
    assert summary["Rows removed by complete-case cleaning"] == 1
