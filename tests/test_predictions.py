import json

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.dummy import DummyClassifier

from src.evaluation import (
    aggregate_fatal_shap_by_source,
    apply_fatal_threshold,
    build_error_cohorts,
    class_pair_overlaps,
    cross_validate_pipeline,
    evaluate_predictions,
    select_fatal_threshold,
    summarize_folds,
)
from src.joint_tuning import (
    fit_parameters_for_alpha,
    render_joint_tuning_markdown,
    sample_joint_candidates,
    select_joint_candidate,
    summarize_joint_candidate,
)
from src.models import (
    OrdinalPredictor,
    cumulative_targets,
    fit_multiclass,
    load_config,
    load_selected_tuned_parameters,
    ordinal_probabilities,
)
from src.preprocessing import build_preprocessor
from src.visualization import PLOT_CLASS_ORDER, save_feature_distribution_by_class
from src.weighting import fine_alpha_grid, interpolated_sample_weight, select_alpha_result


class BinaryModel:
    def __init__(self, positive):
        self.positive = np.asarray(positive)

    def predict_proba(self, X):
        values = self.positive[: len(X)]
        return np.column_stack([1 - values, values])


def test_multiclass_metrics_preserve_fatal_serious_slight_order():
    metrics = evaluate_predictions(np.array([0, 1, 2]), np.array([0, 2, 2]), "test")
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert len(metrics["per_class_f1"]) == 3


def test_fatal_threshold_rule_and_selection():
    probabilities = np.array([[0.4, 0.5, 0.1], [0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    assert apply_fatal_threshold(probabilities, 0.4).tolist() == [0, 2, 0]
    selected, rows = select_fatal_threshold(np.array([0, 2, 0]), probabilities, grid_size=11)
    assert 0 <= selected["threshold"] <= 1
    assert len(rows) == 11


def test_ordinal_targets_and_monotone_probabilities():
    serious, fatal = cumulative_targets(np.array([0, 1, 2]))
    assert serious.tolist() == [1, 1, 0]
    assert fatal.tolist() == [1, 0, 0]
    X = pd.DataFrame({"x": [1, 2]})
    probabilities = ordinal_probabilities(BinaryModel([0.4, 0.8]), BinaryModel([0.7, 0.2]), X)
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1)
    predictor = OrdinalPredictor(BinaryModel([0.4, 0.8]), BinaryModel([0.7, 0.2]))
    assert predictor.predict(X).shape == (2,)


def test_error_cohorts_are_directional():
    cohorts = build_error_cohorts(np.array([0, 1]), np.array([1, 1]))
    assert cohorts["error_type"].tolist() == ["Fatal → Serious", "Correct"]


def test_probability_shape_is_validated():
    with pytest.raises(ValueError, match="shape"):
        apply_fatal_threshold(np.ones((2, 2)), 0.5)


def test_self_contained_pipeline_serializes_and_predicts(tmp_path):
    row = {
        "Road_Type": 1, "Light_Conditions": 1, "Weather_Conditions": 1,
        "Road_Surface_Conditions": 1, "1st_Road_Class": 3, "2nd_Road_Class": -1,
        "Pedestrian_Crossing-Physical_Facilities": 0, "Day_of_Week": 2,
        "Urban_or_Rural_Area": 1, "Season": "Winter", "Speed_limit": 30,
        "hour_sin": 0.0, "hour_cos": 1.0, "rush_hour": 0,
    }
    X = pd.DataFrame([row for _ in range(18)])
    y = np.repeat([0, 1, 2], 6)
    config = {
        "balance": "weighted", "parameters": {"n_estimators": 2, "max_depth": 2}
    }
    path = tmp_path / "model.joblib"
    joblib.dump(fit_multiclass(config, X, y), path)
    probabilities = joblib.load(path).predict_proba(X.iloc[:2])
    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_interpolated_sample_weight_endpoints_and_midpoint():
    y = np.array([0, 1, 1, 2, 2, 2])
    unweighted = interpolated_sample_weight(y, 0.0)
    balanced = interpolated_sample_weight(y, 1.0)
    midpoint = interpolated_sample_weight(y, 0.5)
    assert np.allclose(unweighted, 1.0)
    assert np.allclose(midpoint, 1.0 + 0.5 * (balanced - 1.0))


def test_selected_tuned_parameters_are_loaded_and_validated(tmp_path):
    parameters = {
        "n_estimators": 600,
        "learning_rate": 0.07,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "gamma": 0.1,
        "reg_alpha": 0,
        "reg_lambda": 5,
    }
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps({"selected_parameters": parameters}))

    assert load_selected_tuned_parameters(path) == parameters


def test_selected_tuned_parameters_require_fresh_complete_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="train-tuned"):
        load_selected_tuned_parameters(tmp_path / "missing.json")
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"selected_parameters": {"n_estimators": 600}}))
    with pytest.raises(ValueError, match="missing required"):
        load_selected_tuned_parameters(incomplete)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_selected_tuned_parameters(malformed)


@pytest.mark.parametrize("alpha", [-0.01, 1.01, np.nan])
def test_interpolated_sample_weight_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="between 0 and 1"):
        interpolated_sample_weight(np.array([0, 1, 2]), alpha)


def test_fine_alpha_grid_is_inclusive_and_clipped():
    assert fine_alpha_grid(0.6) == [value / 1000 for value in range(450, 751, 50)]
    assert fine_alpha_grid(0.0) == [value / 1000 for value in range(0, 151, 50)]
    assert fine_alpha_grid(1.0) == [value / 1000 for value in range(850, 1001, 50)]


def test_alpha_selection_uses_macro_f1_then_lower_alpha():
    rows = [
        {"alpha": 0.6, "macro_f1_mean": 0.4},
        {"alpha": 0.4, "macro_f1_mean": 0.4},
        {"alpha": 0.2, "macro_f1_mean": 0.3},
    ]
    assert select_alpha_result(rows)["alpha"] == 0.4


def test_joint_candidate_sampling_is_reproducible_and_within_bounds():
    config = load_config("xgb_joint_tuned")
    first = sample_joint_candidates(config)
    second = sample_joint_candidates(config)
    assert first == second
    assert len(first) == 20
    for candidate in first:
        assert 0.0 <= candidate["alpha"] <= 1.0
        assert candidate["max_depth"] in {3, 4, 5, 6, 7}
        assert 0.02 <= candidate["learning_rate"] <= 0.15
        assert candidate["min_child_weight"] in {1, 3, 5, 8, 12}
        assert 0.65 <= candidate["subsample"] <= 1.0
        assert 0.65 <= candidate["colsample_bytree"] <= 1.0
        assert candidate["gamma"] in {0, 0.25, 0.5, 1, 2}
        assert 0.001 <= candidate["reg_alpha"] <= 5.0
        assert 0.1 <= candidate["reg_lambda"] <= 20.0


def test_joint_candidate_alpha_is_applied_to_only_the_supplied_fold_labels():
    fold_y = np.array([0, 1, 1, 2, 2, 2])
    half = fit_parameters_for_alpha(0.5)(fold_y)["model__sample_weight"]
    full = fit_parameters_for_alpha(1.0)(fold_y)["model__sample_weight"]
    assert len(half) == len(fold_y)
    assert np.allclose(half, 1.0 + 0.5 * (full - 1.0))


def test_joint_candidate_summary_selection_and_report_fields():
    folds = [
        evaluate_predictions(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 1, 2, 2, 2]),
            "joint test",
        )
        for _ in range(3)
    ]
    parameters = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.25,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "n_estimators": 1000,
    }
    row = summarize_joint_candidate(1, 0.4, parameters, folds)
    required = {
        "alpha", "macro_f1_mean", "macro_f1_std", "serious_f1_mean",
        "fatal_precision_mean", "fatal_recall_mean", "fatal_f1_mean", "slight_f1_mean",
    }
    assert required.issubset(row)
    tied_later = {**row, "candidate": 2}
    assert select_joint_candidate([tied_later, row])["candidate"] == 1

    validation = summarize_folds(folds)
    report = {
        "selection_data": "training folds only",
        "candidate_count": 20,
        "search_folds": 3,
        "selected_candidate": row,
        "validation": validation,
        "search": [row],
    }
    markdown = render_joint_tuning_markdown(report)
    assert "Selected candidate: 1" in markdown
    assert "| candidate | alpha |" in markdown


def test_cross_validation_reports_fold_progress():
    messages = []
    X = pd.DataFrame({"feature": range(9)})
    y = np.repeat([0, 1, 2], 3)
    folds, _ = cross_validate_pipeline(
        DummyClassifier(strategy="most_frequent"),
        X,
        y,
        name="progress test",
        n_splits=3,
        progress_callback=messages.append,
    )
    assert len(folds) == 3
    assert sum("started" in message for message in messages) == 3
    assert sum("completed" in message for message in messages) == 3
    assert all("elapsed=" in message for message in messages if "completed" in message)


def test_fatal_shap_is_aggregated_to_interpretable_source_features():
    rows = []
    for road_type in (1, 6):
        rows.append({
            "Road_Type": road_type, "Light_Conditions": 1, "Weather_Conditions": 1,
            "Road_Surface_Conditions": 1, "1st_Road_Class": 3, "2nd_Road_Class": -1,
            "Pedestrian_Crossing-Physical_Facilities": 0, "Day_of_Week": 2,
            "Urban_or_Rural_Area": 1, "Season": "Winter", "Speed_limit": 30,
            "hour_sin": 0.0, "hour_cos": 1.0, "rush_hour": 0,
        })
    preprocessor = build_preprocessor().fit(pd.DataFrame(rows))
    names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    ranking = aggregate_fatal_shap_by_source(
        preprocessor, names, np.ones((3, len(names)))
    )
    by_name = {row["source_feature"]: row for row in ranking}
    assert len(by_name["Road_Type"]["components"]) == 2
    assert by_name["Road_Type"]["mean_abs_fatal_shap"] == pytest.approx(2.0)
    assert set(by_name["Hour_of_Day"]["components"]) == {"hour_sin", "hour_cos"}


def test_pairwise_overlap_uses_shared_support_for_all_true_classes():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    numeric = pd.Series([0, 0, 0, 0, 10, 10])
    rows = class_pair_overlaps(numeric, y_true, categorical=False, bins=2)
    scores = {(row["class_a"], row["class_b"]): row["overlap"] for row in rows}
    assert scores[("Fatal", "Serious")] == pytest.approx(1.0)
    assert scores[("Fatal", "Slight")] == pytest.approx(0.0)

    categorical = pd.Series(["wet", "wet", "wet", "dry", "dry", "dry"])
    rows = class_pair_overlaps(categorical, y_true, categorical=True)
    scores = {(row["class_a"], row["class_b"]): row["overlap"] for row in rows}
    assert scores[("Fatal", "Serious")] == pytest.approx(0.5)


def test_true_class_distribution_plot_uses_requested_order(tmp_path):
    path = tmp_path / "distribution.png"
    save_feature_distribution_by_class(
        pd.Series([20, 30, 40, 50, 60, 70]),
        np.array([2, 2, 1, 1, 0, 0]),
        "Speed_limit",
        0.25,
        path,
    )
    assert PLOT_CLASS_ORDER == ["Slight", "Serious", "Fatal"]
    assert path.stat().st_size > 0
