import pandas as pd

from src.data import deduplicate_raw_records, load_raw


def test_deduplication_ignores_only_exported_row_index():
    raw = pd.DataFrame(
        {
            "Unnamed: 0": [1, 2],
            "Accident_Index": ["collision-a", "collision-a"],
            "Accident_Severity": [3, 3],
            "Longitude": [-0.1, -0.1],
            "Time": [None, None],
        }
    )

    result = deduplicate_raw_records(raw)

    assert len(result) == 1
    assert result.iloc[0]["Unnamed: 0"] == 1


def test_same_identifier_with_substantive_difference_is_not_a_duplicate():
    raw = pd.DataFrame(
        {
            "Unnamed: 0": [1, 2, 3],
            "Accident_Index": ["2.01E+12"] * 3,
            "Accident_Severity": [3, 3, 3],
            "Longitude": [-0.1, -0.2, -0.1],
            "Time": ["08:00", "08:00", "09:00"],
        }
    )

    result = deduplicate_raw_records(raw)

    assert len(result) == 3


def test_difference_in_excluded_model_field_prevents_deduplication():
    raw = pd.DataFrame(
        {
            "Unnamed: 0": [1, 2],
            "Accident_Index": ["collision-a", "collision-a"],
            "Accident_Severity": [3, 3],
            "Longitude": [-0.1, -0.2],
            "Time": ["08:00", "08:00"],
        }
    )

    assert len(deduplicate_raw_records(raw)) == 2


def test_load_raw_deduplicates_before_excluded_fields_are_dropped(tmp_path):
    raw = pd.DataFrame(
        {
            "Unnamed: 0": [1, 2, 3],
            "Accident_Index": ["a", "a", "a"],
            "Accident_Severity": [3, 3, 3],
            "Longitude": [-0.1, -0.1, -0.2],
            "Time": ["08:00", "08:00", "08:00"],
        }
    )
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)

    result = load_raw(path)

    # Rows 1 and 2 differ only by the exported index, while row 3 has a
    # substantive coordinate difference even though coordinates are later dropped.
    assert len(result) == 2
