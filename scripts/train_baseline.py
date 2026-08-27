"""Cross-validate and fit the leakage-safe downsampled XGBoost baseline."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from src import CONFIGS_DIR, MODELS_DIR, RESULTS_DIR
from src.data import load_split
from src.evaluation import cross_validate_pipeline, select_fatal_threshold, upsert_cv_results
from src.models import fit_multiclass, load_config, make_multiclass_pipeline


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config("baseline_xgb")
    X_train, _, y_train, _ = load_split()
    folds, oof = cross_validate_pipeline(
        make_multiclass_pipeline(config), X_train, y_train,
        name=config["name"], n_splits=config["cv_folds"], collect_probabilities=True,
    )
    selected, threshold_rows = select_fatal_threshold(y_train, oof)
    upsert_cv_results(config["name"], config["slug"], folds, RESULTS_DIR / "cv_results.csv")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(threshold_rows).to_csv(
        RESULTS_DIR / "baseline_threshold_results.csv", index=False
    )
    threshold_config = {
        "status": "ready",
        "model_artifact": "models/baseline_xgb.joblib",
        "class_order": ["Fatal", "Serious", "Slight"],
        "fatal_class_index": 0,
        "selection_data": "five-fold out-of-fold training predictions only",
        "selection_metric": "macro_f1",
        "threshold": selected["threshold"],
    }
    (CONFIGS_DIR / "baseline_fatal_threshold.json").write_text(
        json.dumps(threshold_config, indent=2) + "\n"
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_multiclass(config, X_train, y_train), MODELS_DIR / "baseline_xgb.joblib")
    print("[done] models/baseline_xgb.joblib and leakage-safe threshold artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
