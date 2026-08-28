from dataclasses import replace

import pandas as pd
import pytest

from src.data_validation import DatasetContract, sha256_file, validate_dataset


def synthetic_contract(tmp_path):
    rows = []
    for severity in (1, 2, 3):
        for number in range(5):
            rows.append(
                {
                    "Unnamed: 0": len(rows),
                    "Accident_Index": f"{severity}-{number}",
                    "Accident_Severity": severity,
                    "Year": 2014,
                    "Time": "08:00",
                }
            )
    rows.append({**rows[0], "Unnamed: 0": 999})
    raw = pd.DataFrame(rows)
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    contract = DatasetContract(
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        columns=tuple(raw.columns),
        raw_rows=16,
        duplicate_rows=1,
        cleaned_rows=15,
        years=(2014,),
        class_counts={1: 5, 2: 5, 3: 5},
        train_rows=12,
        test_rows=3,
        test_class_counts={1: 1, 2: 1, 3: 1},
    )
    return path, contract


def test_dataset_preflight_reconciles_provenance_cleaning_and_split(tmp_path):
    path, contract = synthetic_contract(tmp_path)
    summary = validate_dataset(path, contract)
    assert summary["duplicate_rows_removed"] == 1
    assert summary["cleaned_rows"] == 15
    assert summary["test_rows"] == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha256", "0" * 64, "SHA-256"),
        ("columns", ("wrong",), "source columns"),
        ("raw_rows", 17, "raw rows"),
        ("years", (2013,), "represented years"),
        ("class_counts", {1: 15}, "severity class counts"),
    ],
)
def test_dataset_preflight_rejects_contract_mismatch(tmp_path, field, value, message):
    path, contract = synthetic_contract(tmp_path)
    with pytest.raises(ValueError, match=message):
        validate_dataset(path, replace(contract, **{field: value}))
