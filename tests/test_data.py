"""Tests for load_processed's balance-variant file selection."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from road_accidents.data import load_processed


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    def _save(stem: str, n: int) -> None:
        pd.DataFrame({"f1": np.arange(n)}).to_parquet(tmp_path / f"X_{stem}.parquet")
        pd.DataFrame({"y": np.arange(n)}).to_parquet(tmp_path / f"y_{stem}.parquet")

    _save("train", 10)  # downsampled train
    _save("train_full", 30)  # full train
    _save("test", 5)
    return tmp_path


def test_downsampled_balance_loads_downsampled_train(processed_dir: Path) -> None:
    X_train, X_test, y_train, y_test = load_processed("downsampled", processed_dir)
    assert len(X_train) == 10
    assert len(y_train) == 10
    assert len(X_test) == 5
    assert len(y_test) == 5


def test_full_balance_loads_full_train(processed_dir: Path) -> None:
    X_train, X_test, y_train, y_test = load_processed("full", processed_dir)
    assert len(X_train) == 30
    assert len(y_train) == 30
    assert len(X_test) == 5
    assert len(y_test) == 5
