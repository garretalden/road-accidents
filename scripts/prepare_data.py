"""End-to-end data preparation: raw CSV → encoded, downsampled train/test parquet.

Also persists the fitted encoders so the Streamlit app can transform new rows
identically to what training saw.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from road_accidents.config import DOWNSAMPLE_TARGETS, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from road_accidents.data import load_raw
from road_accidents.features import add_time_features
from road_accidents.preprocessing import build_preprocessor, transform_to_frame


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] raw CSV")
    df = load_raw()
    print(f"       shape after cleaning: {df.shape}")

    print("[features] time-derived columns")
    df = add_time_features(df)

    X = df.drop(columns="Accident_Severity")
    y = df["Accident_Severity"]

    print("[split] stratified 80/20")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # XGBoost wants class labels in [0, num_class); map 1/2/3 → 0/1/2
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    print("[encode] one-hot categoricals + season, numeric passthrough (fit on train only)")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_raw)
    X_train = transform_to_frame(preprocessor, X_train_raw)
    X_test = transform_to_frame(preprocessor, X_test_raw)
    print(f"       encoded shape: train {X_train.shape}, test {X_test.shape}")

    print("[downsample] Slight/Serious → 60k, Fatal untouched")
    counts = pd.Series(y_train).value_counts()
    sampling_strategy = {**DOWNSAMPLE_TARGETS, 0: int(counts[0])}
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    X_train_down, y_train_down = rus.fit_resample(X_train, y_train)
    print(f"       downsampled: {pd.Series(y_train_down).value_counts().to_dict()}")

    print("[save] parquet (downsampled + full train variants) + preprocessor")
    X_train_down.to_parquet(PROCESSED_DIR / "X_train.parquet")
    pd.DataFrame({"y": y_train_down}).to_parquet(PROCESSED_DIR / "y_train.parquet")
    # Full, non-downsampled train set — for experiments that handle class
    # imbalance via class/sample weighting instead of undersampling.
    X_train.to_parquet(PROCESSED_DIR / "X_train_full.parquet")
    pd.DataFrame({"y": y_train}).to_parquet(PROCESSED_DIR / "y_train_full.parquet")
    X_test.to_parquet(PROCESSED_DIR / "X_test.parquet")
    pd.DataFrame({"y": y_test}).to_parquet(PROCESSED_DIR / "y_test.parquet")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
