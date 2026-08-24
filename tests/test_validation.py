"""Tests for leakage-safe fixed-model validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

import road_accidents.validation as validation
from road_accidents.validation import (
    ValidationSpec,
    out_of_fold_probabilities,
    validate_fixed_model,
)


class RecordingTransformer(TransformerMixin, BaseEstimator):
    fit_indices: list[set[int]] = []

    def fit(self, X, y=None):
        self.__class__.fit_indices.append(set(X.index))
        return self

    def transform(self, X):
        return np.asarray(X[["value"]], dtype=float)


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    fit_weights: list[np.ndarray | None] = []
    predict_indices: list[int] = []

    def fit(self, X, y, sample_weight=None):
        self.__class__.fit_weights.append(
            None if sample_weight is None else np.asarray(sample_weight)
        )
        self.classes_ = np.array([0, 1, 2])
        self.majority_ = int(pd.Series(y).mode().iloc[0])
        return self

    def predict(self, X):
        self.__class__.predict_indices.append(len(X))
        return np.full(len(X), self.majority_, dtype=int)

    def predict_proba(self, X):
        probabilities = np.full((len(X), 3), 0.1)
        probabilities[:, self.majority_] = 0.8
        return probabilities


@pytest.fixture(autouse=True)
def reset_recorders(monkeypatch):
    RecordingTransformer.fit_indices = []
    RecordingClassifier.fit_weights = []
    RecordingClassifier.predict_indices = []
    monkeypatch.setattr(validation, "build_preprocessor", RecordingTransformer)


def _data():
    # Enough examples of every class for three stratified folds and undersampling.
    y_train = np.repeat([0, 1, 2], [18, 36, 66])
    X_train = pd.DataFrame({"value": np.arange(len(y_train))}, index=np.arange(len(y_train)))
    y_test = np.repeat([0, 1, 2], 3)
    X_test = pd.DataFrame(
        {"value": np.arange(len(y_test))}, index=np.arange(1000, 1000 + len(y_test))
    )
    return X_train, y_train, X_test, y_test


def _spec(balance="weighted"):
    return ValidationSpec(
        name="test model",
        slug="test_model",
        balance=balance,
        estimator_factory=RecordingClassifier,
        hyperparameters={},
    )


def test_preprocessor_is_refit_per_fold_and_test_is_never_fitted(monkeypatch):
    X_train, y_train, X_test, y_test = _data()
    final_test_callbacks = []
    result, _ = validate_fixed_model(
        _spec(),
        X_train,
        y_train,
        X_test,
        y_test,
        n_splits=3,
        on_final_test=lambda: final_test_callbacks.append(True),
    )

    assert len(RecordingTransformer.fit_indices) == 4  # three folds + final fit
    assert all(not (indices & set(X_test.index)) for indices in RecordingTransformer.fit_indices)
    assert RecordingTransformer.fit_indices[-1] == set(X_train.index)
    assert final_test_callbacks == [True]
    assert len(result["cv"]["folds"]) == 3
    assert RecordingClassifier.predict_indices[-1] == len(X_test)


def test_folds_are_stratified_and_cover_training_validation_once():
    X_train, y_train, X_test, y_test = _data()
    result, _ = validate_fixed_model(
        _spec(), X_train, y_train, X_test, y_test, n_splits=3
    )

    expected_validation_counts = {"Fatal": 6, "Serious": 12, "Slight": 22}
    for fold in result["cv"]["folds"]:
        assert fold["validation_rows"] == 40
        assert fold["validation_class_counts"] == expected_validation_counts
    assert sum(fold["validation_rows"] for fold in result["cv"]["folds"]) == len(y_train)


def test_weighted_model_computes_balanced_weights_for_every_fit():
    X_train, y_train, X_test, y_test = _data()
    validate_fixed_model(_spec(), X_train, y_train, X_test, y_test, n_splits=3)

    assert len(RecordingClassifier.fit_weights) == 4
    assert all(weights is not None for weights in RecordingClassifier.fit_weights)
    for weights in RecordingClassifier.fit_weights:
        assert len(np.unique(weights)) == 3


def test_baseline_downsamples_only_training_rows_and_does_not_use_weights():
    X_train, y_train, X_test, y_test = _data()
    result, _ = validate_fixed_model(
        _spec("downsampled"),
        X_train,
        y_train,
        X_test,
        y_test,
        n_splits=3,
        downsample_targets={1: 12, 2: 12},
    )

    assert all(weights is None for weights in RecordingClassifier.fit_weights)
    for fold in result["cv"]["folds"]:
        assert fold["fitted_class_counts"] == {"Fatal": 12, "Serious": 12, "Slight": 12}
        assert fold["fitted_rows"] == 36
    assert result["final_fit"]["fitted_class_counts"] == {
        "Fatal": 18,
        "Serious": 12,
        "Slight": 12,
    }


def test_summary_uses_sample_standard_deviation():
    folds = []
    for value in (0.2, 0.4, 0.6):
        folds.append(
            {
                "macro_f1": value,
                "per_class_precision": [value] * 3,
                "per_class_recall": [value] * 3,
                "per_class_f1": [value] * 3,
            }
        )

    summary = validation._summary(folds)
    assert summary["macro_f1"] == pytest.approx({"mean": 0.4, "std": 0.2})
    assert summary["per_class"]["Fatal"]["f1"] == pytest.approx(
        {"mean": 0.4, "std": 0.2}
    )


def test_oof_probabilities_fit_each_fold_once_without_using_held_out_rows():
    X_train, y_train, _, _ = _data()
    probabilities, folds = out_of_fold_probabilities(
        _spec(), X_train, y_train, n_splits=3
    )

    assert probabilities.shape == (len(y_train), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert len(folds) == 3
    assert len(RecordingTransformer.fit_indices) == 3
    assert len(RecordingClassifier.fit_weights) == 3
    assert all(weights is not None for weights in RecordingClassifier.fit_weights)
    assert sum(fold["validation_rows"] for fold in folds) == len(y_train)
