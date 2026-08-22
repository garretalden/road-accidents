"""Run leakage-safe fixed validation for weighted and baseline XGBoost."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from experiments.xgb_weighted_validated import BASELINE_SPEC, WEIGHTED_SPEC
from road_accidents.config import (
    CLASS_NAMES,
    EXPERIMENTS_MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from road_accidents.data import load_raw
from road_accidents.features import add_time_features
from road_accidents.validation import cross_validate_fixed_model, fit_final_model

N_SPLITS = 5
JSON_PATH = REPORTS_DIR / "xgb_validation_results.json"
MARKDOWN_PATH = REPORTS_DIR / "xgb_validation_results.md"


def _markdown(report: dict) -> str:
    lines = [
        "# Leakage-safe XGBoost validation",
        "",
        "Five-fold stratified cross-validation is performed only on the training split. "
        "Each fold fits a fresh preprocessor. The held-out test set is evaluated once after "
        "model selection is complete.",
        "",
        "## Cross-validation summary",
        "",
        "| Model | Macro F1 | Fatal precision | Fatal recall | Fatal F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in report["models"]:
        summary = result["cv"]["summary"]
        fatal = summary["per_class"]["Fatal"]
        lines.append(
            f"| {result['name']} | "
            f"{summary['macro_f1']['mean']:.3f} ± {summary['macro_f1']['std']:.3f} | "
            f"{fatal['precision']['mean']:.3f} ± {fatal['precision']['std']:.3f} | "
            f"{fatal['recall']['mean']:.3f} ± {fatal['recall']['std']:.3f} | "
            f"{fatal['f1']['mean']:.3f} ± {fatal['f1']['std']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-class cross-validation metrics",
            "",
            "| Model | Class | Precision | Recall | F1 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for result in report["models"]:
        per_class = result["cv"]["summary"]["per_class"]
        for class_name in CLASS_NAMES:
            metrics = per_class[class_name]
            lines.append(
                f"| {result['name']} | {class_name} | "
                f"{metrics['precision']['mean']:.3f} ± {metrics['precision']['std']:.3f} | "
                f"{metrics['recall']['mean']:.3f} ± {metrics['recall']['std']:.3f} | "
                f"{metrics['f1']['mean']:.3f} ± {metrics['f1']['std']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Final untouched-test evaluation",
            "",
            "| Model | Macro F1 | Accuracy | Fatal precision | Fatal recall | Fatal F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in report["models"]:
        test = result["test"]
        lines.append(
            f"| {result['name']} | {test['macro_f1']:.3f} | {test['accuracy']:.3f} | "
            f"{test['per_class_precision'][0]:.3f} | {test['per_class_recall'][0]:.3f} | "
            f"{test['per_class_f1'][0]:.3f} |"
        )
    lines.extend(
        [
            "",
            "Classes: 0=Fatal, 1=Serious, 2=Slight. Standard deviations use `ddof=1`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    started = time.time()
    print("[load] raw data and engineer unchanged time features", flush=True)
    df = add_time_features(load_raw())
    X = df.drop(columns="Accident_Severity")
    y_raw = df["Accident_Severity"]

    print("[split] frozen stratified 80/20 train/test split", flush=True)
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, stratify=y_raw, random_state=RANDOM_STATE
    )
    encoder = LabelEncoder().fit(y_train_raw)
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    report = {
        "validation": {
            "n_splits": N_SPLITS,
            "splitter": "StratifiedKFold(shuffle=True)",
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
            "training_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "class_order": CLASS_NAMES,
            "test_policy": "untouched until CV completed for all models",
        },
        "models": [],
    }

    EXPERIMENTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    specs = (BASELINE_SPEC, WEIGHTED_SPEC)
    cv_results = {}
    for spec in specs:
        print(f"[validate] {spec.name}", flush=True)

        def on_fold(number: int, metrics: dict) -> None:
            print(
                f"  fold {number}/{N_SPLITS}: macro-F1={metrics['macro_f1']:.4f}, "
                f"Fatal F1={metrics['per_class_f1'][0]:.4f}",
                flush=True,
            )

        cv_results[spec.slug] = cross_validate_fixed_model(
            spec,
            X_train,
            y_train,
            n_splits=N_SPLITS,
            on_fold=on_fold,
        )

    print("[final] all CV complete; fitting full training sets", flush=True)
    for spec in specs:
        print(f"  fit {spec.name}", flush=True)

        def on_final_test() -> None:
            print("    evaluating untouched test set once", flush=True)

        final_result, pipeline = fit_final_model(
            spec, X_train, y_train, X_test, y_test, on_final_test=on_final_test
        )
        result = {**cv_results[spec.slug], **final_result}
        artifact_path = EXPERIMENTS_MODELS_DIR / f"{spec.slug}.joblib"
        joblib.dump(pipeline, artifact_path)
        result["artifact"] = str(artifact_path.relative_to(Path.cwd()))
        report["models"].append(result)
        print(f"  saved {artifact_path}", flush=True)

    report["elapsed_seconds"] = round(time.time() - started, 1)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")
    MARKDOWN_PATH.write_text(_markdown(report))
    print(f"[done] wrote {JSON_PATH} and {MARKDOWN_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
