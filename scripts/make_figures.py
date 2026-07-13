"""Generate all portfolio figures as PNGs in reports/figures/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from road_accidents.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RAW_CSV_PATH, REPORTS_DIR
from road_accidents.data import load_raw
from road_accidents.features import add_time_features
from road_accidents.viz import (
    save_confusion_matrix,
    save_correlation_heatmap,
    save_lr_coefficient_plot,
    save_rf_permutation_importance,
    save_severity_distribution,
    save_xgb_shap_summary,
)


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[figure] correlation heatmap (from raw features)")
    if RAW_CSV_PATH.exists():
        raw = add_time_features(load_raw())
        save_correlation_heatmap(raw, FIGURES_DIR / "correlation_heatmap.png")
    else:
        print("       skipped — raw CSV missing")

    print("[load] processed data + models")
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["y"].to_numpy()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["y"].to_numpy()

    lr = joblib.load(MODELS_DIR / "lr.joblib")
    rf = joblib.load(MODELS_DIR / "rf.joblib")
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")

    print("[figure] severity class balance")
    save_severity_distribution(y_train, FIGURES_DIR / "severity_distribution.png")

    print("[figure] LR coefficient plot")
    save_lr_coefficient_plot(lr, list(X_train.columns), FIGURES_DIR / "lr_coefficients.png")

    print("[figure] RF permutation importance")
    save_rf_permutation_importance(rf, X_test, y_test, FIGURES_DIR / "rf_permutation.png")

    print("[figure] XGBoost SHAP summary (Fatal class)")
    save_xgb_shap_summary(xgb, X_test, FIGURES_DIR / "xgb_shap_fatal.png")

    results_path = REPORTS_DIR / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        for r in results:
            slug = r["name"].lower().replace(" ", "_")
            print(f"[figure] confusion matrix — {r['name']}")
            save_confusion_matrix(
                r["confusion_matrix"], r["name"], FIGURES_DIR / f"cm_{slug}.png"
            )

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
