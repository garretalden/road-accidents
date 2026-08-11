"""Shared training contract used by both baseline/ and experiments/.

Every trainer — baseline or experimental — takes preprocessed training data
and returns a ``TrainResult`` so ``scripts/train_baseline.py`` and
``scripts/train_experiment.py`` can evaluate and save them identically.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainResult:
    estimator: Any
    best_params: dict
    best_cv_score: float
