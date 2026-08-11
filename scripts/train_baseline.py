"""Train the original class-project models (LR / RF / XGB) on prepared data.

Saves models under models/baseline/ and writes reports/baseline_results.json
+ reports/results.md. See scripts/train_experiment.py for training new models.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from baseline import MODELS
from road_accidents.config import BASELINE_MODELS_DIR, REPORTS_DIR
from road_accidents.data import load_processed
from road_accidents.evaluate import evaluate, write_results_table


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] processed data (downsampled train)")
    X_train, X_test, y_train, y_test = load_processed("downsampled")
    print(f"       train {X_train.shape}, test {X_test.shape}")

    results = []
    for name, (slug, trainer) in MODELS.items():
        print(f"[train] {name}")
        t0 = time.time()
        result = trainer(X_train, y_train)
        elapsed = time.time() - t0
        print(f"       best CV score: {result.best_cv_score:.4f}  ({elapsed:.1f}s)")

        joblib.dump(result.estimator, BASELINE_MODELS_DIR / f"{slug}.joblib")
        metrics = evaluate(result.estimator, X_test, y_test, name)
        metrics["best_params"] = result.best_params
        metrics["best_cv_score"] = float(result.best_cv_score)
        metrics["train_time_seconds"] = round(elapsed, 1)
        results.append(metrics)
        print(f"       test macro-F1: {metrics['macro_f1']:.4f}")

    (REPORTS_DIR / "baseline_results.json").write_text(json.dumps(results, indent=2))
    write_results_table(results, REPORTS_DIR / "results.md")
    print(f"[done] wrote {REPORTS_DIR / 'baseline_results.json'} + {REPORTS_DIR / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
