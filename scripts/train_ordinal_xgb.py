"""Validate and fit two cumulative-binary ordinal XGBoost pipelines."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from sklearn.model_selection import StratifiedKFold
from src import MODELS_DIR, RANDOM_STATE, RESULTS_DIR
from src.data import load_split
from src.evaluation import evaluate_predictions, summarize_folds, upsert_cv_results
from src.models import OrdinalPredictor, fit_ordinal_models, load_config


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config("ordinal_xgb")
    tuning_path = RESULTS_DIR / "xgb_tuning_results.json"
    if not tuning_path.exists():
        raise FileNotFoundError("Run `make train-tuned` before `make train-ordinal`")
    parameters = json.loads(tuning_path.read_text())["selected_parameters"]
    config = {**config, "parameters": parameters}
    X_train, _, y_train, _ = load_split()
    splitter = StratifiedKFold(config["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for number, (fit_index, validation_index) in enumerate(splitter.split(X_train, y_train), start=1):
        serious, fatal = fit_ordinal_models(config, X_train.iloc[fit_index], y_train[fit_index])
        predictor = OrdinalPredictor(serious, fatal)
        metrics = evaluate_predictions(
            y_train[validation_index], predictor.predict(X_train.iloc[validation_index]), config["name"]
        )
        metrics["fold"] = number
        folds.append(metrics)
        print(f"[fold {number}/{config['cv_folds']}] macro-F1={metrics['macro_f1']:.4f}")
    upsert_cv_results(config["name"], config["slug"], folds, RESULTS_DIR / "cv_results.csv")
    serious, fatal = fit_ordinal_models(config, X_train, y_train)
    ordinal_dir = MODELS_DIR / "ordinal"
    ordinal_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(serious, ordinal_dir / "serious_or_worse.joblib")
    joblib.dump(fatal, ordinal_dir / "fatal.joblib")
    result = {
        "selection_data": "training folds only", "parameters": parameters,
        "validation": summarize_folds(folds),
        "artifacts": ["models/ordinal/serious_or_worse.joblib", "models/ordinal/fatal.joblib"],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ordinal_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
