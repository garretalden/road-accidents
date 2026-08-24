"""Tune, validate, threshold, and fit class-weighted XGBoost on training data only."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from sklearn.model_selection import ParameterSampler
from src import CONFIGS_DIR, MODELS_DIR, RANDOM_STATE, RESULTS_DIR
from src.data import load_split
from src.evaluation import cross_validate_pipeline, select_fatal_threshold, summarize_folds, upsert_cv_results
from src.models import balanced_fit_parameters, fit_multiclass, load_config, make_multiclass_pipeline


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config("tuned_xgb")
    X_train, _, y_train, _ = load_split()
    candidates = list(ParameterSampler(
        config["search_space"], n_iter=config["candidate_count"], random_state=RANDOM_STATE
    ))
    search_results = []
    for number, parameters in enumerate(candidates, start=1):
        candidate = {**config, "parameters": {**config["parameters"], **parameters}}
        folds, _ = cross_validate_pipeline(
            make_multiclass_pipeline(candidate), X_train, y_train, name=config["name"],
            n_splits=config["search_folds"], fit_parameters=balanced_fit_parameters,
        )
        summary = summarize_folds(folds)
        search_results.append({"candidate": number, "parameters": candidate["parameters"], "cv": summary})
        print(f"[search {number}/{len(candidates)}] macro-F1={summary['macro_f1']['mean']:.4f}")
    winner = max(search_results, key=lambda row: row["cv"]["macro_f1"]["mean"])
    selected_config = {**config, "parameters": winner["parameters"]}
    folds, oof = cross_validate_pipeline(
        make_multiclass_pipeline(selected_config), X_train, y_train, name=config["name"],
        n_splits=config["validation_folds"], fit_parameters=balanced_fit_parameters,
        collect_probabilities=True,
    )
    selected, threshold_rows = select_fatal_threshold(y_train, oof)
    upsert_cv_results(config["name"], config["slug"], folds, RESULTS_DIR / "cv_results.csv")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "selection_data": "training folds only", "winner_candidate": winner["candidate"],
        "selected_parameters": winner["parameters"], "search": search_results,
        "validation": summarize_folds(folds),
    }
    (RESULTS_DIR / "xgb_tuning_results.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(threshold_rows).to_csv(RESULTS_DIR / "threshold_results.csv", index=False)
    threshold_config = {
        "status": "ready", "model_artifact": "models/tuned_xgb.joblib",
        "class_order": ["Fatal", "Serious", "Slight"], "fatal_class_index": 0,
        "selection_data": "five-fold out-of-fold training predictions only",
        "selection_metric": "macro_f1", "threshold": selected["threshold"],
    }
    (CONFIGS_DIR / "fatal_threshold.json").write_text(json.dumps(threshold_config, indent=2) + "\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_multiclass(selected_config, X_train, y_train), MODELS_DIR / "tuned_xgb.joblib")
    print("[done] models/tuned_xgb.joblib and leakage-safe threshold artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
