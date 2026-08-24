"""Tests for cumulative-binary ordinal accident severity modeling."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

import road_accidents.ordinal as ordinal
from road_accidents.ordinal import (
    _fit_ordinal,
    cumulative_targets,
    cumulative_to_class_probabilities,
    enforce_cumulative_order,
)


def test_cumulative_targets_follow_fatal_serious_slight_class_encoding():
    at_least_serious, fatal = cumulative_targets(np.array([0, 1, 2, 0, 2]))
    assert at_least_serious.tolist() == [1, 1, 0, 1, 0]
    assert fatal.tolist() == [1, 0, 0, 1, 0]


def test_order_projection_uses_closest_monotone_pair():
    serious_plus, fatal = enforce_cumulative_order(
        np.array([0.8, 0.2]), np.array([0.1, 0.6])
    )
    assert serious_plus.tolist() == pytest.approx([0.8, 0.4])
    assert fatal.tolist() == pytest.approx([0.1, 0.4])
    assert np.all(fatal <= serious_plus)


def test_class_probabilities_are_ordered_nonnegative_and_sum_to_one():
    probabilities = cumulative_to_class_probabilities(
        np.array([0.8, 0.2]), np.array([0.1, 0.6])
    )
    # Output columns retain the repository's Fatal, Serious, Slight encoding.
    assert np.allclose(probabilities, [[0.1, 0.7, 0.2], [0.4, 0.0, 0.6]])
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


class PassthroughTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X[["value"]], dtype=float)


class RecordingBinaryClassifier:
    fits = []

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, X, y, sample_weight=None):
        self.__class__.fits.append((np.asarray(y), np.asarray(sample_weight)))
        self.positive_probability = float(np.average(y, weights=sample_weight))
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), self.positive_probability)
        return np.column_stack([1 - positive, positive])


def test_binary_tasks_compute_their_own_balanced_sample_weights(monkeypatch):
    RecordingBinaryClassifier.fits = []
    monkeypatch.setattr(ordinal, "build_preprocessor", PassthroughTransformer)
    y = np.array([0, 1, 1, 2, 2, 2, 2, 2])
    X = pd.DataFrame({"value": np.arange(len(y))})

    pipeline, metadata = _fit_ordinal(
        X, y, {"n_estimators": 1}, estimator_factory=RecordingBinaryClassifier
    )

    assert len(RecordingBinaryClassifier.fits) == 2
    serious_target, serious_weights = RecordingBinaryClassifier.fits[0]
    fatal_target, fatal_weights = RecordingBinaryClassifier.fits[1]
    assert serious_target.tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert fatal_target.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    for target, weights in ((serious_target, serious_weights), (fatal_target, fatal_weights)):
        assert weights[target == 0].sum() == pytest.approx(weights[target == 1].sum())
    assert metadata["training_rows"] == len(y)
    assert pipeline.predict_proba(X).shape == (len(y), 3)
