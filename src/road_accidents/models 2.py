"""Model training routines for Logistic Regression, Random Forest, XGBoost.

Each ``train_*`` function accepts already-preprocessed training data and
returns ``(fitted_estimator, best_params, best_cv_score)``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE


@dataclass
class TrainResult:
    estimator: Any
    best_params: dict
    best_cv_score: float


def train_lr(X_train: pd.DataFrame, y_train: np.ndarray) -> TrainResult:
    """Multinomial logistic regression with GridSearchCV over C.

    Uses a Pipeline so the fitted scaler travels with the estimator and the
    Streamlit app can predict from raw features without a separate scaler file.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])
    param_grid = {"lr__C": [0.001, 0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    return TrainResult(search.best_estimator_, search.best_params_, search.best_score_)


def train_rf(X_train: pd.DataFrame, y_train: np.ndarray) -> TrainResult:
    """Random forest with RandomizedSearchCV over depth/leaf/features/class weight."""
    rf = RandomForestClassifier(
        n_estimators=150,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    param_distributions = {
        "max_depth": [15, 25, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.5],
        "class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return TrainResult(search.best_estimator_, search.best_params_, search.best_score_)


def train_xgb(X_train: pd.DataFrame, y_train: np.ndarray) -> TrainResult:
    """XGBoost with RandomizedSearchCV over depth/subsample/learning-rate/trees."""
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
    search.fit(X_train, y_train)
    return TrainResult(search.best_estimator_, search.best_params_, search.best_score_)
