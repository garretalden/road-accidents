"""Generate the tuned-XGBoost error analysis and portfolio modeling report."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from src import CONFIGS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RESULTS_DIR
from src.data import load_split
from src.evaluation import build_error_cohorts, evaluate_predictions, histogram_overlap
from src.models import load_config
from src.visualization import (
    compute_tree_shap, save_confusion_matrix, save_fatal_precision_recall_curve,
    save_fatal_shap, save_fatal_threshold_tradeoff, save_feature_distribution_overlap,
    save_global_shap_summary, save_normalized_confusion_matrix,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    model_path = MODELS_DIR / "tuned_xgb.joblib"
    threshold_path = CONFIGS_DIR / "fatal_threshold.json"
    tradeoff_path = RESULTS_DIR / "threshold_results.csv"
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    required = [model_path, threshold_path, tradeoff_path, comparison_path]
    missing = [str(path.relative_to(Path.cwd())) for path in required if not path.exists()]
    threshold_config = json.loads(threshold_path.read_text()) if threshold_path.exists() else {}
    if missing or threshold_config.get("status") != "ready":
        raise FileNotFoundError(
            "Complete `make train-tuned` and `make evaluate` first; missing or stale: "
            f"{missing or ['configs/fatal_threshold.json']}"
        )
    model = joblib.load(model_path)
    _, X_test, _, y_test = load_split()
    probabilities = model.predict_proba(X_test)
    predictions = np.argmax(probabilities, axis=1)
    metrics = evaluate_predictions(y_test, predictions, load_config("tuned_xgb")["name"])
    cohorts = build_error_cohorts(y_test, predictions)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    distribution_dir = FIGURES_DIR / "feature_distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(metrics["confusion_matrix"], metrics["name"], FIGURES_DIR / "confusion_matrix.png")
    save_normalized_confusion_matrix(
        metrics["confusion_matrix"], metrics["name"], FIGURES_DIR / "confusion_matrix_normalized.png"
    )
    threshold = float(threshold_config["threshold"])
    pr = save_fatal_precision_recall_curve(
        y_test, probabilities[:, 0], threshold, FIGURES_DIR / "fatal_precision_recall.png"
    )
    threshold_rows = pd.read_csv(tradeoff_path).to_dict(orient="records")
    save_fatal_threshold_tradeoff(threshold_rows, threshold, FIGURES_DIR / "fatal_threshold_tradeoff.png")
    X_shap, shap_values = compute_tree_shap(model, X_test)
    save_global_shap_summary(shap_values, list(X_shap.columns), FIGURES_DIR / "shap_global.png")
    save_fatal_shap(shap_values, X_shap, FIGURES_DIR / "shap_fatal.png")
    importance = np.abs(shap_values[:, :, 0]).mean(axis=0)
    top_features = [X_shap.columns[index] for index in np.argsort(importance)[::-1][:5]]
    correct = cohorts["is_correct"].to_numpy()
    transformed_all = pd.DataFrame(
        model.named_steps["preprocessor"].transform(X_test),
        columns=[name.split("__", 1)[-1] for name in model.named_steps["preprocessor"].get_feature_names_out()],
        index=X_test.index,
    )
    overlap_rows = []
    for feature in top_features:
        overlap = histogram_overlap(transformed_all.loc[correct, feature], transformed_all.loc[~correct, feature])
        filename = f"{_slug(feature)}.png"
        save_feature_distribution_overlap(
            transformed_all[feature], correct, feature, overlap, distribution_dir / filename
        )
        overlap_rows.append((feature, overlap, filename))
    comparison = pd.read_csv(comparison_path)
    lines = [
        "# UK road-accident severity modeling report", "",
        "## Executive summary", "",
        "This project predicts Fatal, Serious, or Slight collision severity using only information "
        "available before a collision. All reported estimates use fold-local preprocessing; model "
        "selection and Fatal-threshold selection use training data only.", "",
        "## Model comparison", "", comparison.to_markdown(index=False, floatfmt=".3f"), "",
        "![Model comparison](figures/model_comparison.png)", "", "## Tuned XGBoost error analysis", "",
        f"Held-out macro F1: **{metrics['macro_f1']:.3f}**; Fatal precision/recall/F1: "
        f"**{metrics['per_class_precision'][0]:.3f} / {metrics['per_class_recall'][0]:.3f} / "
        f"{metrics['per_class_f1'][0]:.3f}**.", "",
        "![Confusion matrix](figures/confusion_matrix.png)", "",
        "![Normalized confusion matrix](figures/confusion_matrix_normalized.png)", "",
        "## Fatal operating point", "",
        f"The frozen threshold **{threshold:.4f}** was selected from OOF training predictions. On "
        f"the untouched test set it yields precision **{pr['precision_at_selected_threshold']:.3f}** "
        f"and recall **{pr['recall_at_selected_threshold']:.3f}**.", "",
        "![Fatal precision-recall](figures/fatal_precision_recall.png)", "",
        "![Fatal threshold tradeoff](figures/fatal_threshold_tradeoff.png)", "",
        "## Model interpretation", "",
        "![Global SHAP](figures/shap_global.png)", "",
        "![Fatal SHAP](figures/shap_fatal.png)", "", "### Correct-versus-error feature overlap", "",
    ]
    for feature, overlap, filename in overlap_rows:
        lines += [f"- `{feature}` overlap: **{overlap:.3f}**", f"  ![{feature}](figures/feature_distributions/{filename})", ""]
    lines += [
        "## Limitations", "",
        "The data contains reported personal-injury collisions rather than all road exposure. "
        "Weather and road-surface inputs assume contemporaneous observations at a known location. "
        "Predicted probabilities should not be interpreted as calibrated risk without a separate "
        "calibration study.", "",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "modeling_report.md").write_text("\n".join(lines))
    print("[done] reports/modeling_report.md and portfolio figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
