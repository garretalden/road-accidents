"""Cross-validate and fit the fixed class-weighted XGBoost model."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from src import MODELS_DIR, RESULTS_DIR
from src.data import load_split
from src.evaluation import cross_validate_pipeline, upsert_cv_results
from src.models import balanced_fit_parameters, fit_multiclass, load_config, make_multiclass_pipeline


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config("weighted_xgb")
    X_train, _, y_train, _ = load_split()
    folds, _ = cross_validate_pipeline(
        make_multiclass_pipeline(config), X_train, y_train,
        name=config["name"], n_splits=config["cv_folds"],
        fit_parameters=balanced_fit_parameters,
    )
    upsert_cv_results(config["name"], config["slug"], folds, RESULTS_DIR / "cv_results.csv")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_multiclass(config, X_train, y_train), MODELS_DIR / "weighted_xgb.joblib")
    print("[done] models/weighted_xgb.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
