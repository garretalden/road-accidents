"""Train LR / RF / XGB on prepared data. Save models + a Markdown results table + JSON."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from road_accidents.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from road_accidents.evaluate import evaluate, write_results_table
from road_accidents.models import train_lr, train_rf, train_xgb

TRAINERS = {
    "Logistic Regression": ("lr", train_lr),
    "Random Forest": ("rf", train_rf),
    "XGBoost": ("xgb", train_xgb),
}


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] processed data")
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["y"].to_numpy()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["y"].to_numpy()
    print(f"       train {X_train.shape}, test {X_test.shape}")

    results = []
    for name, (slug, trainer) in TRAINERS.items():
        print(f"[train] {name}")
        t0 = time.time()
        result = trainer(X_train, y_train)
        elapsed = time.time() - t0
        print(f"       best CV score: {result.best_cv_score:.4f}  ({elapsed:.1f}s)")

        joblib.dump(result.estimator, MODELS_DIR / f"{slug}.joblib")
        metrics = evaluate(result.estimator, X_test, y_test, name)
        metrics["best_params"] = result.best_params
        metrics["best_cv_score"] = float(result.best_cv_score)
        metrics["train_time_seconds"] = round(elapsed, 1)
        results.append(metrics)
        print(f"       test macro-F1: {metrics['macro_f1']:.4f}")

    (REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    write_results_table(results, REPORTS_DIR / "results.md")
    print(f"[done] wrote {REPORTS_DIR / 'results.json'} + {REPORTS_DIR / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
