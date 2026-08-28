"""Validate the exact audited source and final modeling cohort before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import RAW_DATA_PATH
from src.data_validation import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    args = parser.parse_args()
    summary = validate_dataset(args.data)
    print(
        "[preflight] validated audited dataset: "
        f"{summary['raw_rows']:,} raw, "
        f"{summary['duplicate_rows_removed']:,} duplicates removed, "
        f"{summary['cleaned_rows']:,} cleaned, "
        f"{summary['train_rows']:,}/{summary['test_rows']:,} train/test"
    )
    print(f"[preflight] SHA-256 {summary['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
