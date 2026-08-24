"""Validate and evaluate a cumulative-binary ordinal XGBoost experiment."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from road_accidents.config import (
    CLASS_NAMES,
    EXPERIMENTS_MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from road_accidents.data import load_raw
from road_accidents.features import add_time_features
from road_accidents.ordinal import (
    NAME,
    SLUG,
    cross_validate_ordinal_xgb,
    fit_final_ordinal_xgb,
)

N_SPLITS = 5
TUNING_JSON = REPORTS_DIR / "xgb_tuning_results.json"
OUTPUT_JSON = REPORTS_DIR / "xgb_ordinal_results.json"
OUTPUT_MD = REPORTS_DIR / "xgb_ordinal_results.md"
ARTIFACT = EXPERIMENTS_MODELS_DIR / f"{SLUG}.joblib"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_multiclass_hashes() -> dict[str, str]:
    paths = [
        path
        for path in (EXPERIMENTS_MODELS_DIR.parent / "baseline").glob("*.joblib")
    ] + [
        path
        for path in EXPERIMENTS_MODELS_DIR.glob("*.joblib")
        if path != ARTIFACT
    ]
    return {str(path.relative_to(Path.cwd())): _sha256(path) for path in sorted(paths)}


def _markdown(report: dict) -> str:
    ordinal = report["ordinal"]
    cv = ordinal["cv"]
    lines = [
        "# Ordinal cumulative XGBoost results",
        "",
        "Two full-data, class-weighted binary XGBoost models estimate `P(Y >= Serious)` "
        "and `P(Y = Fatal)`. Each fold fits a fresh copy of the existing preprocessor. "
        "Independent cumulative probabilities are projected to the constraint "
        "`P(Fatal) <= P(Y >= Serious)` before class probabilities are derived.",
        "",
        "## Binary five-fold validation",
        "",
        "| Binary task | Precision | Recall | F1 | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "at_least_serious": "Serious or Fatal vs Slight",
        "fatal": "Fatal vs Serious or Slight",
    }
    for task, label in labels.items():
        metrics = cv["binary_tasks"][task]
        lines.append(
            f"| {label} | {metrics['precision']['mean']:.3f} ± {metrics['precision']['std']:.3f} | "
            f"{metrics['recall']['mean']:.3f} ± {metrics['recall']['std']:.3f} | "
            f"{metrics['f1']['mean']:.3f} ± {metrics['f1']['std']:.3f} | "
            f"{metrics['accuracy']['mean']:.3f} ± {metrics['accuracy']['std']:.3f} |"
        )

    lines += [
        "",
        "## Combined ordinal five-fold validation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Macro F1 | {cv['summary']['macro_f1']['mean']:.3f} ± {cv['summary']['macro_f1']['std']:.3f} |",
        "",
        "| Class | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for class_name in CLASS_NAMES:
        metrics = cv["summary"]["per_class"][class_name]
        lines.append(
            f"| {class_name} | {metrics['precision']['mean']:.3f} ± {metrics['precision']['std']:.3f} | "
            f"{metrics['recall']['mean']:.3f} ± {metrics['recall']['std']:.3f} | "
            f"{metrics['f1']['mean']:.3f} ± {metrics['f1']['std']:.3f} |"
        )
    ordering = cv["ordering"]
    lines += [
        "",
        f"Raw cumulative-order violations corrected: {ordering['raw_violation_count']:,} "
        f"of {ordering['validation_rows']:,} OOF predictions "
        f"({ordering['raw_violation_count'] / ordering['validation_rows']:.2%}).",
        "",
        "## Final untouched-test comparison",
        "",
        "| Model | Macro F1 | Accuracy | Fatal P/R/F1 | Serious P/R/F1 | Slight P/R/F1 |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for model in report["test_comparison"]:
        test = model["test"]
        cells = []
        for index in range(len(CLASS_NAMES)):
            cells.append(
                f"{test['per_class_precision'][index]:.3f} / "
                f"{test['per_class_recall'][index]:.3f} / "
                f"{test['per_class_f1'][index]:.3f}"
            )
        lines.append(
            f"| {model['name']} | {test['macro_f1']:.3f} | {test['accuracy']:.3f} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} |"
        )
    lines += [
        "",
        "No thresholds were tuned. Binary probabilities use the classifiers' native "
        "outputs; final class predictions use argmax after monotone projection.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    started = time.time()
    if not TUNING_JSON.exists():
        raise FileNotFoundError("Run `make tune-xgb` before the ordinal experiment")
    tuning_report = json.loads(TUNING_JSON.read_text())
    parameters = tuning_report["search"]["selected_parameters"]
    hashes_before = _existing_multiclass_hashes()

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

    print("[validate] five fold-local ordinal fits on training data only", flush=True)

    def progress(number: int, metrics: dict) -> None:
        serious = metrics["binary_tasks"]["at_least_serious"]["f1"]
        fatal = metrics["binary_tasks"]["fatal"]["f1"]
        print(
            f"  fold {number}/{N_SPLITS}: combined macro-F1={metrics['macro_f1']:.4f}, "
            f"binary F1 serious+={serious:.4f}, fatal={fatal:.4f}",
            flush=True,
        )

    cv = cross_validate_ordinal_xgb(
        X_train, y_train, parameters, n_splits=N_SPLITS, on_fold=progress
    )

    print("[final] ordinal method fixed; fitting both models on all training rows", flush=True)
    final, pipeline = fit_final_ordinal_xgb(
        X_train,
        y_train,
        X_test,
        y_test,
        parameters,
        on_final_test=lambda: print("  evaluating untouched test set once", flush=True),
    )
    EXPERIMENTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT)
    ordinal = {**cv, **final, "artifact": str(ARTIFACT.relative_to(Path.cwd()))}
    tuned_reference = next(
        model for model in tuning_report["comparison"] if model["slug"] == "xgb_weighted_tuned"
    )
    report = {
        "validation": {
            "n_splits": N_SPLITS,
            "splitter": "StratifiedKFold(shuffle=True)",
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
            "training_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "class_order": CLASS_NAMES,
            "test_policy": "untouched until ordinal CV and probability conversion were fixed",
        },
        "method": {
            "targets": {
                "at_least_serious": "Slight=0; Serious or Fatal=1",
                "fatal": "Slight or Serious=0; Fatal=1",
            },
            "weighting": "balanced sample weights computed separately per binary fit",
            "ordering_projection": "if P(Fatal) > P(Y>=Serious), replace both with their mean",
            "class_probabilities": {
                "Fatal": "P(Fatal)",
                "Serious": "P(Y>=Serious) - P(Fatal)",
                "Slight": "1 - P(Y>=Serious)",
            },
            "decision": "argmax class probability; no tuned thresholds",
        },
        "ordinal": ordinal,
        "test_comparison": [
            {
                "name": tuned_reference["name"],
                "slug": tuned_reference["slug"],
                "test": tuned_reference["test"],
            },
            {"name": NAME, "slug": SLUG, "test": ordinal["test"]},
        ],
        "protected_multiclass_artifact_hashes": hashes_before,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if _existing_multiclass_hashes() != hashes_before:
        raise RuntimeError("An existing multiclass model artifact changed")
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    OUTPUT_MD.write_text(_markdown(report))
    print(f"[done] wrote {ARTIFACT}, {OUTPUT_JSON}, and {OUTPUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
