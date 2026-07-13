"""Plot generation. All functions save PNGs headlessly."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from .config import CLASS_NAMES, RANDOM_STATE  # noqa: E402


def save_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix (numeric features)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_severity_distribution(y: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=y, ax=ax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f"{i}: {n}" for i, n in enumerate(CLASS_NAMES)])
    ax.set_xlabel("Accident severity")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion_matrix(cm: list[list[int]], name: str, path: Path) -> None:
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_lr_coefficient_plot(lr_pipeline: Any, feature_names: list[str], path: Path) -> None:
    """Top-10 features by mean absolute coefficient across classes.

    Sign is taken from the class where the feature's magnitude is largest,
    so the bar direction is meaningful for interpretation.
    """
    lr = lr_pipeline.named_steps["lr"] if hasattr(lr_pipeline, "named_steps") else lr_pipeline
    coef = lr.coef_
    importance = np.mean(np.abs(coef), axis=0)
    top_idx = np.argsort(importance)[-10:]
    top_features = np.array(feature_names)[top_idx]
    sign_source = np.argmax(np.abs(coef[:, top_idx]), axis=0)
    signed = np.array([coef[sign_source[i], top_idx[i]] for i in range(10)])

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["green" if v < 0 else "red" for v in signed]
    ax.barh(top_features, signed, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Top 10 Logistic Regression Coefficients (signed)")
    ax.set_xlabel("Coefficient value (impact on log-odds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_rf_permutation_importance(
    rf: Any, X_test: pd.DataFrame, y_test: np.ndarray, path: Path, sample_size: int = 10_000
) -> None:
    """Permutation importance on a subset of the test set (for speed)."""
    n = min(sample_size, len(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_perm = X_test.iloc[idx]
    y_perm = y_test[idx]

    result = permutation_importance(
        rf, X_perm, y_perm, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    top_idx = np.argsort(result.importances_mean)[-10:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.array(X_test.columns)[top_idx], result.importances_mean[top_idx])
    ax.set_title("Permutation Feature Importance (Random Forest)")
    ax.set_xlabel("Decrease in score when feature is shuffled")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_xgb_shap_summary(
    xgb: Any, X_test: pd.DataFrame, path: Path, n_rows: int = 2000, target_class: int = 0
) -> None:
    """SHAP bar summary for a specific class (default: Fatal)."""
    X_sample = X_test.iloc[:n_rows, :]
    explainer = shap.TreeExplainer(xgb.get_booster())
    shap_values = explainer.shap_values(X_sample)
    class_shap = shap_values[:, :, target_class]

    fig = plt.figure(figsize=(12, 6))
    shap.summary_plot(
        class_shap,
        X_sample,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title(f"SHAP Feature Importance (XGBoost, class = {CLASS_NAMES[target_class]})")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
