"""Template for a new experiment.

Copy this file to ``experiments/<your_module>.py`` (drop the leading
underscore — the runner skips modules named ``_*``) and fill in ``NAME``,
``SLUG``, and ``train()``. Then run::

    make experiment NAME=<your_module>

This trains on ``data/processed/X_train.parquet`` (same preprocessed data
baseline uses), evaluates on the held-out test set with
``road_accidents.evaluate.evaluate()``, saves the fitted estimator to
``models/experiments/<SLUG>.joblib``, and upserts the result into
``reports/experiments_results.json`` — baseline's results are never touched.
"""

import numpy as np
import pandas as pd

from road_accidents.training import TrainResult

NAME = "My New Model"  # display name shown in results tables
SLUG = "my_new_model"  # filesystem-safe id — used for the joblib filename


def train(X_train: pd.DataFrame, y_train: np.ndarray) -> TrainResult:
    """Fit your model and return it wrapped in a TrainResult.

    ``best_params``/``best_cv_score`` can be empty/NaN if you're not doing a
    hyperparameter search — they're just recorded for the results table.
    """
    raise NotImplementedError("copy this file and implement train()")
