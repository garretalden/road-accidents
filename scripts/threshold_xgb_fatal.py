"""Select a Fatal probability threshold from tuned-XGBoost OOF predictions."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from experiments.xgb_weighted_tuned import make_tuned_spec
from road_accidents.config import (
    CLASS_NAMES,
    EXPERIMENTS_MODELS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from road_accidents.data import load_raw
from road_accidents.evaluate import evaluate_predictions
from road_accidents.features import add_time_features
from road_accidents.thresholding import apply_fatal_threshold, search_fatal_thresholds
from road_accidents.validation import out_of_fold_probabilities

N_SPLITS = 5
TUNING_JSON = REPORTS_DIR / "xgb_tuning_results.json"
MODEL_PATH = EXPERIMENTS_MODELS_DIR / "xgb_weighted_tuned.joblib"
THRESHOLD_PATH = EXPERIMENTS_MODELS_DIR / "xgb_weighted_tuned_fatal_threshold.json"
OUTPUT_JSON = REPORTS_DIR / "xgb_fatal_threshold_results.json"
OUTPUT_MD = REPORTS_DIR / "xgb_fatal_threshold_results.md"
FIGURE_PATH = FIGURES_DIR / "xgb_fatal_threshold_tradeoff.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_fingerprint(parameters: dict) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_frozen_model(pipeline, parameters: dict) -> None:
    model_parameters = pipeline.named_steps["model"].get_params()
    mismatches = {
        key: {"report": value, "model": model_parameters.get(key)}
        for key, value in parameters.items()
        if model_parameters.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Tuned report and saved model parameters disagree: {mismatches}")


def _plot(results: list[dict], macro_threshold: float, fatal_threshold: float) -> None:
    thresholds = np.array([result["threshold"] for result in results])
    precision = np.array([result["fatal_precision"] for result in results])
    recall = np.array([result["fatal_recall"] for result in results])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scatter = axes[0].scatter(recall, precision, c=thresholds, cmap="viridis", s=18)
    axes[0].set(title="Fatal precision-recall tradeoff", xlabel="Fatal recall", ylabel="Fatal precision")
    fig.colorbar(scatter, ax=axes[0], label="Fatal threshold")

    for metric, label in (
        ("fatal_precision", "Fatal precision"),
        ("fatal_recall", "Fatal recall"),
        ("fatal_f1", "Fatal F1"),
        ("macro_f1", "Macro F1"),
        ("predicted_fatal_proportion", "Predicted Fatal proportion"),
    ):
        axes[1].plot(thresholds, [row[metric] for row in results], label=label)
    axes[1].axvline(macro_threshold, color="black", linestyle="--", label="Macro-F1 choice")
    if fatal_threshold != macro_threshold:
        axes[1].axvline(fatal_threshold, color="gray", linestyle=":", label="Fatal-F1 choice")
    axes[1].set(title="Validation metrics by threshold", xlabel="Fatal threshold", ylabel="Metric")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _summary_rows(search: dict) -> list[dict]:
    chosen = {
        search["selected_for_macro_f1"]["threshold"],
        search["selected_for_fatal_f1"]["threshold"],
    }
    targets = chosen | {0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75}
    rows = search["thresholds"]
    selected = {
        min(rows, key=lambda row: abs(row["threshold"] - target))["threshold"]
        for target in targets
    }
    return [row for row in rows if row["threshold"] in selected]


def _markdown(report: dict) -> str:
    macro = report["search"]["selected_for_macro_f1"]
    fatal = report["search"]["selected_for_fatal_f1"]
    lines = [
        "# Tuned weighted XGBoost Fatal-threshold results",
        "",
        "The threshold was selected exclusively from five-fold out-of-fold probabilities "
        "on the training split. The saved tuned XGBoost model was not changed.",
        "",
        "## Selected validation thresholds",
        "",
        "| Selection | Threshold | Macro F1 | Fatal precision | Fatal recall | Fatal F1 | Predicted Fatal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in (("Maximum macro F1", macro), ("Maximum Fatal F1", fatal)):
        lines.append(
            f"| {label} | {row['threshold']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['fatal_precision']:.4f} | {row['fatal_recall']:.4f} | "
            f"{row['fatal_f1']:.4f} | {row['predicted_fatal_proportion']:.2%} |"
        )

    lines += [
        "",
        "## Validation threshold summary",
        "",
        "| Threshold | Macro F1 | Fatal precision | Fatal recall | Fatal F1 | Predicted Fatal |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _summary_rows(report["search"]):
        lines.append(
            f"| {row['threshold']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['fatal_precision']:.4f} | {row['fatal_recall']:.4f} | "
            f"{row['fatal_f1']:.4f} | {row['predicted_fatal_proportion']:.2%} |"
        )

    lines += [
        "",
        f"![Fatal threshold tradeoff]({FIGURE_PATH.relative_to(REPORTS_DIR)})",
        "",
        "## Final untouched-test comparison",
        "",
        "| Model | Macro F1 | Accuracy | Fatal precision | Fatal recall | Fatal F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in report["test_comparison"]:
        test = model["test"]
        lines.append(
            f"| {model['name']} | {test['macro_f1']:.3f} | {test['accuracy']:.3f} | "
            f"{test['per_class_precision'][0]:.3f} | {test['per_class_recall'][0]:.3f} | "
            f"{test['per_class_f1'][0]:.3f} |"
        )
    lines += ["", "Classes: 0=Fatal, 1=Serious, 2=Slight.", ""]
    return "\n".join(lines)


def main() -> int:
    started = time.time()
    if not TUNING_JSON.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("Run `make tune-xgb` before threshold tuning")

    tuning_report = json.loads(TUNING_JSON.read_text())
    parameters = tuning_report["search"]["selected_parameters"]
    model_hash_before = _sha256(MODEL_PATH)
    saved_pipeline = joblib.load(MODEL_PATH)
    _validate_frozen_model(saved_pipeline, parameters)

    print("[load] raw data and recreate frozen 80/20 split", flush=True)
    df = add_time_features(load_raw())
    X = df.drop(columns="Accident_Severity")
    y = df["Accident_Severity"]
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    encoder = LabelEncoder().fit(y_train_raw)
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    print("[oof] fit frozen tuned specification across five training folds", flush=True)
    spec = make_tuned_spec(parameters)

    def progress(number: int, metadata: dict) -> None:
        print(
            f"  fold {number}/{N_SPLITS}: predicted {metadata['validation_rows']:,} held-out rows",
            flush=True,
        )

    oof_probabilities, folds = out_of_fold_probabilities(
        spec, X_train, y_train, n_splits=N_SPLITS, on_fold=progress
    )
    print("[select] sweep Fatal thresholds using OOF predictions only", flush=True)
    search = search_fatal_thresholds(y_train, oof_probabilities)
    selected = search["selected_for_macro_f1"]
    fatal_best = search["selected_for_fatal_f1"]

    threshold_config = {
        "model_artifact": str(MODEL_PATH.relative_to(Path.cwd())),
        "model_sha256": model_hash_before,
        "parameter_fingerprint": _parameter_fingerprint(parameters),
        "class_order": CLASS_NAMES,
        "fatal_class_index": 0,
        "selection_data": "five-fold out-of-fold predictions on training split only",
        "selection_metric": "macro_f1",
        "threshold": selected["threshold"],
        "rule": "Fatal if P(Fatal) >= threshold; otherwise argmax(Serious, Slight)",
    }
    EXPERIMENTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(json.dumps(threshold_config, indent=2) + "\n")
    print(f"[freeze] threshold={selected['threshold']:.4f}; wrote {THRESHOLD_PATH}", flush=True)

    print("[test] apply frozen threshold once with the existing saved model", flush=True)
    test_probabilities = saved_pipeline.predict_proba(X_test)
    threshold_predictions = apply_fatal_threshold(test_probabilities, selected["threshold"])
    threshold_test = evaluate_predictions(
        y_test, threshold_predictions, "XGBoost (class-weighted, tuned + Fatal threshold)"
    )
    reference_models = [
        {"name": model["name"], "slug": model["slug"], "test": model["test"]}
        for model in tuning_report["comparison"]
    ]
    threshold_model = {
        "name": threshold_test["name"],
        "slug": "xgb_weighted_tuned_fatal_threshold",
        "threshold": selected["threshold"],
        "test": threshold_test,
    }
    report = {
        "validation": {
            "n_splits": N_SPLITS,
            "splitter": "StratifiedKFold(shuffle=True)",
            "random_state": RANDOM_STATE,
            "training_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "class_order": CLASS_NAMES,
            "threshold_selection_policy": "OOF training predictions only",
            "folds": folds,
        },
        "frozen_model": {
            "artifact": str(MODEL_PATH.relative_to(Path.cwd())),
            "sha256": model_hash_before,
            "parameters": parameters,
            "parameter_fingerprint": _parameter_fingerprint(parameters),
        },
        "search": search,
        "chosen_threshold": threshold_config,
        "fatal_f1_alternative": fatal_best,
        "test_comparison": [*reference_models, threshold_model],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if _sha256(MODEL_PATH) != model_hash_before:
        raise RuntimeError("The frozen tuned model artifact changed during threshold tuning")

    _plot(
        search["thresholds"],
        selected["threshold"],
        fatal_best["threshold"],
    )
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    OUTPUT_MD.write_text(_markdown(report))
    print(f"[done] wrote {OUTPUT_JSON}, {OUTPUT_MD}, and {FIGURE_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
