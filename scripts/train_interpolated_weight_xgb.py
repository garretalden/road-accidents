"""Select class-weight interpolation by two-stage CV, then fit XGBoost."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src import MODELS_DIR, RESULTS_DIR
from src.data import load_split
from src.evaluation import cross_validate_pipeline, summarize_folds, upsert_cv_results
from src.models import fit_multiclass, interpolated_fit_parameters, load_config, make_multiclass_pipeline
from src.weighting import fine_alpha_grid, select_alpha_result


def progress(message: str) -> None:
    print(message, flush=True)


def summarize_alpha(stage: str, alpha: float, folds: list[dict]) -> dict:
    summary = summarize_folds(folds)
    per_class = summary["per_class"]
    return {
        "stage": stage,
        "alpha": alpha,
        "macro_f1_mean": summary["macro_f1"]["mean"],
        "macro_f1_std": summary["macro_f1"]["std"],
        "fatal_precision_mean": per_class["Fatal"]["precision"]["mean"],
        "fatal_recall_mean": per_class["Fatal"]["recall"]["mean"],
        "fatal_f1_mean": per_class["Fatal"]["f1"]["mean"],
        "serious_f1_mean": per_class["Serious"]["f1"]["mean"],
        "slight_f1_mean": per_class["Slight"]["f1"]["mean"],
    }


def evaluate_grid(config: dict, X_train: pd.DataFrame, y_train, stage: str, alphas: list[float]):
    rows = []
    folds_by_alpha = {}
    for number, alpha in enumerate(alphas, start=1):
        started = perf_counter()
        progress(f"[{stage} {number}/{len(alphas)}] alpha={alpha:.3f} started")
        candidate = {**config, "weight_alpha": alpha}
        folds, _ = cross_validate_pipeline(
            make_multiclass_pipeline(candidate),
            X_train,
            y_train,
            name=config["name"],
            n_splits=config["cv_folds"],
            fit_parameters=lambda fold_y, value=alpha: interpolated_fit_parameters(fold_y, value),
            progress_callback=lambda message, prefix=stage, value=alpha: progress(
                f"[{prefix} alpha={value:.3f}] {message}"
            ),
        )
        row = summarize_alpha(stage, alpha, folds)
        rows.append(row)
        folds_by_alpha[alpha] = folds
        progress(
            f"[{stage} {number}/{len(alphas)}] alpha={alpha:.3f} "
            f"macro-F1={row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} "
            f"elapsed={perf_counter() - started:.1f}s"
        )
    return rows, folds_by_alpha


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config("interpolated_weight_xgb")
    progress(
        f"[setup] loading training data for {config['cv_folds']}-fold CV; "
        f"coarse alphas={config['coarse_alphas']}"
    )
    load_started = perf_counter()
    X_train, _, y_train, _ = load_split()
    progress(
        f"[setup] loaded {len(X_train):,} training rows "
        f"in {perf_counter() - load_started:.1f}s"
    )

    progress("[coarse] search started")
    coarse_rows, _ = evaluate_grid(
        config, X_train, y_train, "coarse", config["coarse_alphas"]
    )
    coarse_winner = select_alpha_result(coarse_rows)
    fine_alphas = fine_alpha_grid(
        coarse_winner["alpha"], radius=config["fine_radius"], step=config["fine_step"]
    )
    progress(
        f"[coarse] winner alpha={coarse_winner['alpha']:.3f}; "
        f"fine alphas={fine_alphas}"
    )
    progress("[fine] search started")
    fine_rows, fine_folds = evaluate_grid(config, X_train, y_train, "fine", fine_alphas)
    winner = select_alpha_result(fine_rows)
    progress(
        f"[fine] selected alpha={winner['alpha']:.3f} "
        f"with macro-F1={winner['macro_f1_mean']:.4f}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = [*coarse_rows, *fine_rows]
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "xgb_weight_alpha_search.csv", index=False)
    report = {
        "selection_data": "training folds only",
        "selection_metric": "mean three-fold macro_f1",
        "tie_break": "lower alpha",
        "weight_formula": "1 + alpha * (balanced_weight - 1)",
        "fixed_parameters": config["parameters"],
        "coarse_winner": coarse_winner,
        "fine_grid": fine_alphas,
        "selected_alpha": winner["alpha"],
        "selected_result": winner,
        "search": all_rows,
    }
    (RESULTS_DIR / "xgb_weight_alpha_search.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    upsert_cv_results(
        config["name"], config["slug"], fine_folds[winner["alpha"]],
        RESULTS_DIR / "cv_results.csv",
    )

    selected_config = {**config, "weight_alpha": winner["alpha"]}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    progress("[fit] fitting selected model on the full training split")
    fit_started = perf_counter()
    joblib.dump(
        fit_multiclass(selected_config, X_train, y_train),
        MODELS_DIR / "interpolated_weight_xgb.joblib",
    )
    progress(
        "[done] models/interpolated_weight_xgb.joblib "
        f"with selected alpha={winner['alpha']:.3f}; "
        f"full fit elapsed={perf_counter() - fit_started:.1f}s; "
        "reports written to reports/results/xgb_weight_alpha_search.csv and .json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
