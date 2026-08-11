"""Train + evaluate one experiment module, e.g.:

    uv run python scripts/train_experiment.py my_new_model

Loads experiments/<module_name>.py, trains it on the same preprocessed data
baseline uses, evaluates on the held-out test set, saves the model to
models/experiments/<SLUG>.joblib, and upserts the result (by slug) into
reports/experiments_results.json without touching any other results.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from road_accidents.config import EXPERIMENTS_MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from road_accidents.evaluate import evaluate


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: train_experiment.py <module_name>  (a file in experiments/)")
        return 1
    module_name = sys.argv[1]
    if module_name.startswith("_"):
        print(f"error: {module_name!r} looks like a template/private module, not an experiment")
        return 1

    module = importlib.import_module(f"experiments.{module_name}")
    name = module.NAME
    slug = module.SLUG

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] processed data")
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["y"].to_numpy()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["y"].to_numpy()
    print(f"       train {X_train.shape}, test {X_test.shape}")

    print(f"[train] {name}")
    t0 = time.time()
    result = module.train(X_train, y_train)
    elapsed = time.time() - t0
    print(f"       best CV score: {result.best_cv_score:.4f}  ({elapsed:.1f}s)")

    joblib.dump(result.estimator, EXPERIMENTS_MODELS_DIR / f"{slug}.joblib")
    metrics = evaluate(result.estimator, X_test, y_test, name)
    metrics["slug"] = slug
    metrics["best_params"] = result.best_params
    metrics["best_cv_score"] = float(result.best_cv_score)
    metrics["train_time_seconds"] = round(elapsed, 1)
    print(f"       test macro-F1: {metrics['macro_f1']:.4f}")

    results_path = REPORTS_DIR / "experiments_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    results = [r for r in results if r.get("slug") != slug]
    results.append(metrics)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
