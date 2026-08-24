import numpy as np
import pandas as pd
import pytest
import joblib

from src.evaluation import apply_fatal_threshold, build_error_cohorts, evaluate_predictions, select_fatal_threshold
from src.models import OrdinalPredictor, cumulative_targets, fit_multiclass, ordinal_probabilities


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
