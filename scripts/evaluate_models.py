"""Evaluate all completed model families once on the frozen held-out test split."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from src import FIGURES_DIR, MODELS_DIR, RESULTS_DIR
from src.data import load_split
from src.evaluation import evaluate
from src.models import OrdinalPredictor, load_config
from src.visualization import save_model_comparison, save_severity_distribution


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    required = {
        "baseline_xgb": MODELS_DIR / "baseline_xgb.joblib",
        "weighted_xgb": MODELS_DIR / "weighted_xgb.joblib",
        "tuned_xgb": MODELS_DIR / "tuned_xgb.joblib",
        "ordinal_serious": MODELS_DIR / "ordinal" / "serious_or_worse.joblib",
        "ordinal_fatal": MODELS_DIR / "ordinal" / "fatal.joblib",
    }
    missing = [str(path.relative_to(Path.cwd())) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Retrain before evaluation; missing artifacts: {missing}")
    _, X_test, _, y_test = load_split()
    models = [
        (load_config("baseline_xgb")["name"], joblib.load(required["baseline_xgb"])),
        (load_config("weighted_xgb")["name"], joblib.load(required["weighted_xgb"])),
        (load_config("tuned_xgb")["name"], joblib.load(required["tuned_xgb"])),
        (load_config("ordinal_xgb")["name"], OrdinalPredictor(
            joblib.load(required["ordinal_serious"]), joblib.load(required["ordinal_fatal"])
        )),
    ]
    results = [evaluate(model, X_test, y_test, name) for name, model in models]
    rows = []
    for result in results:
        rows.append({
            "model": result["name"], "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "fatal_precision": result["per_class_precision"][0],
            "fatal_recall": result["per_class_recall"][0],
            "fatal_f1": result["per_class_f1"][0],
            "serious_f1": result["per_class_f1"][1], "slight_f1": result["per_class_f1"][2],
        })
    frame = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    save_model_comparison(frame, FIGURES_DIR / "model_comparison.png")
    save_severity_distribution(y_test, FIGURES_DIR / "class_distribution.png")
    ordinal = next(result for result in results if result["name"] == load_config("ordinal_xgb")["name"])
    existing = json.loads((RESULTS_DIR / "ordinal_results.json").read_text())
    existing["untouched_test"] = ordinal
    (RESULTS_DIR / "ordinal_results.json").write_text(json.dumps(existing, indent=2) + "\n")
    print("[done] held-out model comparison written once")
    return 0


if __name__ == "__main__":
    sys.exit(main())
