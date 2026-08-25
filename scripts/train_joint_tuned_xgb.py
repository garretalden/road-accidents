"""Jointly tune XGBoost parameters and class-weight interpolation on training folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from src import MODELS_DIR, RESULTS_DIR
from src.data import load_split
from src.evaluation import cross_validate_pipeline, summarize_folds, upsert_cv_results
from src.joint_tuning import (
    fit_parameters_for_alpha,
    render_joint_tuning_markdown,
    sample_joint_candidates,
    select_joint_candidate,
    summarize_joint_candidate,
)
from src.models import fit_multiclass, load_config, make_multiclass_pipeline


def progress(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config("xgb_joint_tuned")
    progress("[setup] loading the frozen training split")
    load_started = perf_counter()
    X_train, _, y_train, _ = load_split()
    progress(f"[setup] loaded {len(X_train):,} training rows in {perf_counter() - load_started:.1f}s")

    candidates = sample_joint_candidates(config)
    progress(
        f"[search] sampled {len(candidates)} candidates with random_state=42; "
        f"running {config['search_folds']}-fold CV"
    )
    search_rows = []
    for number, sampled in enumerate(candidates, start=1):
        candidate_started = perf_counter()
        alpha = float(sampled["alpha"])
        parameters = {**config["parameters"], **{k: v for k, v in sampled.items() if k != "alpha"}}
        candidate_config = {**config, "parameters": parameters, "weight_alpha": alpha}
        progress(
            f"[candidate {number}/{len(candidates)}] started alpha={alpha:.6f} "
            f"parameters={parameters}"
        )
        folds, _ = cross_validate_pipeline(
            make_multiclass_pipeline(candidate_config),
            X_train,
            y_train,
            name=config["name"],
            n_splits=config["search_folds"],
            fit_parameters=fit_parameters_for_alpha(alpha),
            progress_callback=lambda message, candidate=number: progress(
                f"[candidate {candidate}/{len(candidates)}] {message}"
            ),
        )
        row = summarize_joint_candidate(number, alpha, parameters, folds)
        search_rows.append(row)
        progress(
            f"[candidate {number}/{len(candidates)}] completed "
            f"macro-F1={row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} "
            f"elapsed={perf_counter() - candidate_started:.1f}s"
        )

    winner = select_joint_candidate(search_rows)
    alpha = winner["alpha"]
    selected_parameters = {
        name: winner[name] for name in [*config["search_space"].keys(), "n_estimators"]
        if name != "alpha"
    }
    selected_config = {**config, "parameters": selected_parameters, "weight_alpha": alpha}
    progress(
        f"[selection] candidate={winner['candidate']} alpha={alpha:.6f} "
        f"macro-F1={winner['macro_f1_mean']:.4f}; configuration frozen"
    )

    progress(f"[validation] starting fresh {config['validation_folds']}-fold validation")
    validation_started = perf_counter()
    validation_folds, _ = cross_validate_pipeline(
        make_multiclass_pipeline(selected_config),
        X_train,
        y_train,
        name=config["name"],
        n_splits=config["validation_folds"],
        fit_parameters=fit_parameters_for_alpha(alpha),
        progress_callback=lambda message: progress(f"[validation] {message}"),
    )
    validation = summarize_folds(validation_folds)
    progress(
        f"[validation] completed macro-F1={validation['macro_f1']['mean']:.4f} "
        f"± {validation['macro_f1']['std']:.4f} "
        f"elapsed={perf_counter() - validation_started:.1f}s"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    upsert_cv_results(
        config["name"], config["slug"], validation_folds, RESULTS_DIR / "cv_results.csv"
    )
    report = {
        "selection_data": "training folds only",
        "selection_metric": "mean three-fold macro_f1",
        "tie_break": "earlier deterministic sample order",
        "weight_formula": "1 + alpha * (balanced_weight - 1)",
        "random_state": 42,
        "candidate_count": config["candidate_count"],
        "search_folds": config["search_folds"],
        "validation_folds": config["validation_folds"],
        "search_space": config["search_space"],
        "selected_candidate": winner,
        "selected_parameters": selected_parameters,
        "selected_alpha": alpha,
        "validation": validation,
        "search": search_rows,
    }
    json_path = RESULTS_DIR / "xgb_joint_tuning_results.json"
    markdown_path = RESULTS_DIR / "xgb_joint_tuning_results.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(render_joint_tuning_markdown(report))

    model_path = MODELS_DIR / "experiments" / "xgb_joint_tuned.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    progress("[fit] fitting the frozen winner on the complete training split")
    fit_started = perf_counter()
    joblib.dump(fit_multiclass(selected_config, X_train, y_train), model_path)
    progress(
        f"[done] fitted {model_path.relative_to(Path.cwd())} in "
        f"{perf_counter() - fit_started:.1f}s"
    )
    progress(
        "[done] search and validation reports written to "
        "reports/results/xgb_joint_tuning_results.json and .md"
    )
    progress("[next] run `make evaluate` once to evaluate the untouched test split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
