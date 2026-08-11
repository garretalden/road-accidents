"""XGBoost trained on the full (non-downsampled) train set, using per-sample
class weights instead of undersampling to handle the Fatal/Serious/Slight
imbalance.

Same search space as ``baseline/models.py::train_xgb`` for a fair comparison
against baseline XGBoost — the only difference is the train set (full vs.
downsampled) and the added ``sample_weight``. XGBoost has no multiclass
``class_weight`` constructor argument, so per-sample weights via
``compute_sample_weight`` are the equivalent for boosted trees.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from road_accidents.config import RANDOM_STATE
from road_accidents.training import TrainResult

NAME = "XGBoost (class-weighted)"
SLUG = "xgb_class_weighted"
BALANCE = "full"


def train(X_train: pd.DataFrame, y_train: np.ndarray) -> TrainResult:
    """XGBoost with RandomizedSearchCV over depth/subsample/learning-rate/trees,
    fit with balanced per-sample weights instead of undersampling."""
    sample_weight = compute_sample_weight("balanced", y_train)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        eval_metric="mlogloss",
        learning_rate=0.05,
        n_estimators=150,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_distributions = {
        "max_depth": [3, 4, 5, 6, 7],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "learning_rate": [0.03, 0.05, 0.07],
        "n_estimators": [120, 150, 200],
    }
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train, sample_weight=sample_weight)
    return TrainResult(search.best_estimator_, search.best_params_, search.best_score_)
