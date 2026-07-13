"""Download the UK road accidents CSV via the Kaggle API.

If Kaggle credentials aren't present, prints setup instructions and exits nonzero.
Idempotent: skips download if the file already exists at the expected path.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accidents.config import RAW_CSV_PATH

DATASET = "silicon99/dft-accident-data"
CSV_FILENAME = "UK_Accident.csv"


def main() -> int:
    if RAW_CSV_PATH.exists():
        size_mb = RAW_CSV_PATH.stat().st_size / (1024 * 1024)
        print(f"[skip] {RAW_CSV_PATH} already exists ({size_mb:.0f} MB).")
        return 0

    if not (Path.home() / ".kaggle" / "kaggle.json").exists() and "KAGGLE_KEY" not in os.environ:
        print(
            "Kaggle credentials not found.\n"
            "Options:\n"
            f"  1. Manually download {CSV_FILENAME} from kaggle.com/datasets/{DATASET}\n"
            f"     and place it at {RAW_CSV_PATH}.\n"
            "  2. Set up the Kaggle API: create a token at kaggle.com/settings/account\n"
            "     and save it to ~/.kaggle/kaggle.json (chmod 600), then rerun.",
            file=sys.stderr,
        )
        return 1

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "The `kaggle` package isn't installed. Run `uv sync --group dev` "
            "and retry, or download the CSV manually.",
            file=sys.stderr,
        )
        return 1

    api = KaggleApi()
    api.authenticate()

    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {DATASET} → {RAW_CSV_PATH.parent}")
    api.dataset_download_files(DATASET, path=str(RAW_CSV_PATH.parent), quiet=False, unzip=False)

    zip_path = next(RAW_CSV_PATH.parent.glob("*.zip"), None)
    if zip_path is None:
        print("Download completed but no zip found — inspect data/raw/.", file=sys.stderr)
        return 1

    print(f"[extract] {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_CSV_PATH.parent)
    zip_path.unlink()

    if not RAW_CSV_PATH.exists():
        # Some Kaggle uploads have a different casing / spacing — look for the CSV.
        csvs = list(RAW_CSV_PATH.parent.glob("*.csv"))
        if len(csvs) == 1:
            csvs[0].rename(RAW_CSV_PATH)
        else:
            print(
                f"Expected {RAW_CSV_PATH.name} after extraction. Found: "
                f"{[p.name for p in csvs]}. Rename manually.",
                file=sys.stderr,
            )
            return 1

    print(f"[done] {RAW_CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
