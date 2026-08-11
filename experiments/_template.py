"""Template for a new experiment.

Copy this file to ``experiments/<your_module>.py`` (drop the leading
underscore — the runner skips modules named ``_*``) and fill in ``NAME``,
``SLUG``, and ``train()``. Then run::

    make experiment NAME=<your_module>

This evaluates on the held-out test set with
``road_accidents.evaluate.evaluate()``, saves the fitted estimator to
``models/experiments/<SLUG>.joblib``, and upserts the result into
``reports/experiments_results.json`` — baseline's results are never touched.

By default it trains on the same downsampled train set baseline uses. Set
``BALANCE = "full"`` below to instead train on the full, non-downsampled train
set and handle class imbalance via weighting inside ``train()`` — e.g.
``class_weight="balanced"`` for Logistic Regression / Random Forest, or
``sample_weight=sklearn.utils.class_weight.compute_sample_weight("balanced",
y_train)`` passed to ``fit()`` for XGBoost, which has no multiclass
``class_weight`` constructor argument. See ``experiments/xgb_class_weighted.py``
for a worked example.
"""

import numpy as np
import pandas as pd

from road_accidents.training import TrainResult

NAME = "My New Model"  # display name shown in results tables
SLUG = "my_new_model"  # filesystem-safe id — used for the joblib filename
# BALANCE = "full"  # uncomment to train on the full (non-downsampled) train set


def train(X_train: pd.DataFrame, y_train: np.ndarray) -> TrainResult:
    """Fit your model and return it wrapped in a TrainResult.

    ``best_params``/``best_cv_score`` can be empty/NaN if you're not doing a
    hyperparameter search — they're just recorded for the results table.
    """
    raise NotImplementedError("copy this file and implement train()")
