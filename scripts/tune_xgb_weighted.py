"""Tune weighted XGBoost on training folds, then evaluate the final test once."""

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

from experiments.xgb_weighted_tuned import make_tuned_spec
from road_accidents.config import CLASS_NAMES, EXPERIMENTS_MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from road_accidents.data import load_raw
from road_accidents.features import add_time_features
from road_accidents.tuning import generate_candidates, tune_weighted_xgb
from road_accidents.validation import cross_validate_fixed_model, fit_final_model

OUTPUT_JSON = REPORTS_DIR / "xgb_tuning_results.json"
OUTPUT_MD = REPORTS_DIR / "xgb_tuning_results.md"
REFERENCE_JSON = REPORTS_DIR / "xgb_validation_results.json"


def _markdown(report: dict) -> str:
    lines = [
        "# Weighted XGBoost tuning results",
        "",
        "Hyperparameters were selected using three-fold stratified CV on training data only. "
        "The winner was then evaluated with five-fold CV and fitted on all training data before "
        "the final test set was evaluated once.",
        "",
        "## Five-fold CV comparison",
        "",
        "| Model | Macro F1 | Fatal precision | Fatal recall | Fatal F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in report["comparison"]:
        s = model["cv"]["summary"]
        f = s["per_class"]["Fatal"]
        lines.append(
            f"| {model['name']} | {s['macro_f1']['mean']:.3f} ± {s['macro_f1']['std']:.3f} | {f['precision']['mean']:.3f} ± {f['precision']['std']:.3f} | {f['recall']['mean']:.3f} ± {f['recall']['std']:.3f} | {f['f1']['mean']:.3f} ± {f['f1']['std']:.3f} |"
        )
    lines += [
        "",
        "## Five-fold per-class metrics",
        "",
        "| Model | Class | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model in report["comparison"]:
        for name in CLASS_NAMES:
            m = model["cv"]["summary"]["per_class"][name]
            lines.append(
                f"| {model['name']} | {name} | {m['precision']['mean']:.3f} ± {m['precision']['std']:.3f} | {m['recall']['mean']:.3f} ± {m['recall']['std']:.3f} | {m['f1']['mean']:.3f} ± {m['f1']['std']:.3f} |"
            )
    lines += [
        "",
        "## Final untouched-test comparison",
        "",
        "| Model | Macro F1 | Accuracy | Fatal precision | Fatal recall | Fatal F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in report["comparison"]:
        t = model["test"]
        lines.append(
            f"| {model['name']} | {t['macro_f1']:.3f} | {t['accuracy']:.3f} | {t['per_class_precision'][0]:.3f} | {t['per_class_recall'][0]:.3f} | {t['per_class_f1'][0]:.3f} |"
        )
    p = report["search"]["selected_parameters"]
    lines += [
        "",
        "## Selected configuration",
        "",
        f"Candidate {report['search']['winner_candidate']} won by mean macro F1. Final tree count is the median early-stopped tree count across its search folds.",
        "",
        "```json",
        json.dumps(p, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    started = time.time()
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError("Run `make validate-xgb` before tuning for comparison results")
    reference = json.loads(REFERENCE_JSON.read_text())
    print("[load] raw data and frozen 80/20 split", flush=True)
    df = add_time_features(load_raw())
    X = df.drop(columns="Accident_Severity")
    y = df["Accident_Severity"]
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    encoder = LabelEncoder().fit(y_train_raw)
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    candidates = generate_candidates(12)
    print("[search] 12 candidates × 3 stratified folds", flush=True)

    def progress(candidate, fold, metrics):
        print(
            f"  fold {fold}/3 candidate {candidate}/12: macro-F1={metrics['macro_f1']:.4f}, trees={metrics['selected_trees']}",
            flush=True,
        )

    search = tune_weighted_xgb(X_train, y_train, candidates=candidates, on_result=progress)
    spec = make_tuned_spec(search["selected_parameters"])
    print(
        f"[winner] candidate {search['winner_candidate']}: {search['selected_parameters']}",
        flush=True,
    )
    print("[validate] winner with established five-fold framework", flush=True)

    def fold_progress(number, metrics):
        print(
            f"  fold {number}/5: macro-F1={metrics['macro_f1']:.4f}, Fatal F1={metrics['per_class_f1'][0]:.4f}",
            flush=True,
        )

    tuned_cv = cross_validate_fixed_model(spec, X_train, y_train, n_splits=5, on_fold=fold_progress)
    print("[final] tuning and CV complete; fitting full training data", flush=True)
    final, pipeline = fit_final_model(
        spec,
        X_train,
        y_train,
        X_test,
        y_test,
        on_final_test=lambda: print("  evaluating untouched test set once", flush=True),
    )
    tuned = {**tuned_cv, **final}
    EXPERIMENTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = EXPERIMENTS_MODELS_DIR / "xgb_weighted_tuned.joblib"
    joblib.dump(pipeline, artifact)
    tuned["artifact"] = str(artifact.relative_to(Path.cwd()))
    report = {
        "search": search,
        "comparison": [*reference["models"], tuned],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    OUTPUT_MD.write_text(_markdown(report))
    print(f"[done] wrote {artifact}, {OUTPUT_JSON}, and {OUTPUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
