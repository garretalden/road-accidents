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
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from . import CLASS_NAMES, RANDOM_STATE  # noqa: E402


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


def save_normalized_confusion_matrix(cm: list[list[int]], name: str, path: Path) -> None:
    """Save a row-normalized confusion matrix."""
    values = np.asarray(cm, dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Normalized confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison(results: pd.DataFrame, path: Path) -> None:
    """Compare held-out macro F1 and Fatal F1 across models."""
    melted = results.melt(
        id_vars="model",
        value_vars=["macro_f1", "fatal_f1"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="score", y="model", hue="metric", ax=ax)
    ax.set(xlim=(0, 1), xlabel="Held-out test score", ylabel="")
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def compute_tree_shap(
    pipeline: Any,
    X: pd.DataFrame,
    *,
    n_rows: int = 2000,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return deterministic transformed rows and multiclass TreeSHAP values."""
    size = min(n_rows, len(X))
    sample = X.sample(n=size, random_state=RANDOM_STATE).sort_index()
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(sample)
    names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    frame = pd.DataFrame(transformed, columns=names, index=sample.index)
    values = shap.TreeExplainer(pipeline.named_steps["model"]).shap_values(frame)
    if isinstance(values, list):
        array = np.stack([np.asarray(item) for item in values], axis=2)
    else:
        array = np.asarray(values)
        if array.ndim == 3 and array.shape[0] == len(CLASS_NAMES):
            array = np.moveaxis(array, 0, 2)
    expected = (len(frame), frame.shape[1], len(CLASS_NAMES))
    if array.shape != expected:
        raise ValueError(f"unexpected SHAP shape {array.shape}; expected {expected}")
    return frame, array


def save_fatal_shap(shap_values: np.ndarray, X: pd.DataFrame, path: Path) -> None:
    """Save one Fatal-specific SHAP beeswarm."""
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_values[:, :, 0], X, max_display=15, show=False)
    plt.title("Fatal-specific SHAP effects")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_global_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    path: Path,
    *,
    max_display: int = 15,
) -> None:
    """Plot mean absolute SHAP importance across every output class."""
    if shap_values.ndim != 3 or shap_values.shape[2] != len(CLASS_NAMES):
        raise ValueError("shap_values must have shape (rows, features, classes)")
    per_class = np.abs(shap_values).mean(axis=0)
    top = np.argsort(per_class.mean(axis=1))[-max_display:]
    names = np.asarray(feature_names)[top]
    values = per_class[top]

    fig, ax = plt.subplots(figsize=(11, 7))
    left = np.zeros(len(top))
    for class_index, class_name in enumerate(CLASS_NAMES):
        ax.barh(names, values[:, class_index], left=left, label=class_name)
        left += values[:, class_index]
    ax.set_title("Global SHAP importance across severity classes")
    ax.set_xlabel("Mean |SHAP value| (stacked across classes)")
    ax.legend(title="Model output")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_fatal_precision_recall_curve(
    y_true: np.ndarray,
    fatal_probabilities: np.ndarray,
    selected_threshold: float,
    path: Path,
) -> dict[str, float]:
    """Plot the held-out Fatal one-vs-rest precision-recall curve."""
    fatal_true = np.asarray(y_true) == 0
    scores = np.asarray(fatal_probabilities, dtype=float)
    precision, recall, _ = precision_recall_curve(fatal_true, scores)
    selected = scores >= selected_threshold
    true_positives = int(np.sum(selected & fatal_true))
    selected_precision = true_positives / int(selected.sum()) if selected.any() else 0.0
    selected_recall = true_positives / int(fatal_true.sum()) if fatal_true.any() else 0.0
    average_precision = float(average_precision_score(fatal_true, scores))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"Test PR curve (AP = {average_precision:.3f})")
    ax.scatter(
        [selected_recall],
        [selected_precision],
        color="red",
        zorder=3,
        label=f"Frozen threshold = {selected_threshold:.4f}",
    )
    ax.axhline(float(fatal_true.mean()), color="gray", linestyle="--", label="Fatal prevalence")
    ax.set(
        title="Fatal precision–recall curve (held-out test set)",
        xlabel="Fatal recall",
        ylabel="Fatal precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "average_precision": average_precision,
        "selected_threshold": float(selected_threshold),
        "precision_at_selected_threshold": selected_precision,
        "recall_at_selected_threshold": selected_recall,
    }


def save_fatal_threshold_tradeoff(
    threshold_results: list[dict], selected_threshold: float, path: Path
) -> None:
    """Plot training-OOF Fatal metrics over the threshold sweep."""
    thresholds = np.asarray([row["threshold"] for row in threshold_results])
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, label in (
        ("fatal_precision", "Fatal precision"),
        ("fatal_recall", "Fatal recall"),
        ("fatal_f1", "Fatal F1"),
        ("macro_f1", "Macro F1"),
    ):
        ax.plot(thresholds, [row[key] for row in threshold_results], label=label)
    ax.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"Selected = {selected_threshold:.4f}",
    )
    ax.set(
        title="Fatal threshold tradeoff (training OOF predictions)",
        xlabel="Fatal probability threshold",
        ylabel="Metric",
        xlim=(float(thresholds.min()), float(thresholds.max())),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_feature_distribution_overlap(
    values: pd.Series,
    is_correct: np.ndarray,
    feature: str,
    overlap: float,
    path: Path,
) -> None:
    """Plot correct-versus-incorrect feature distributions."""
    frame = pd.DataFrame(
        {
            feature: values.to_numpy(),
            "Prediction": np.where(is_correct, "Correct", "Incorrect"),
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    if pd.api.types.is_numeric_dtype(values):
        unique = values.nunique(dropna=True)
        if unique <= 12:
            sns.histplot(
                data=frame,
                x=feature,
                hue="Prediction",
                stat="probability",
                common_norm=False,
                discrete=True,
                multiple="dodge",
                shrink=0.8,
                ax=ax,
            )
        else:
            sns.histplot(
                data=frame,
                x=feature,
                hue="Prediction",
                stat="density",
                common_norm=False,
                element="step",
                fill=False,
                bins=20,
                ax=ax,
            )
    else:
        proportions = (
            frame.groupby("Prediction")[feature]
            .value_counts(normalize=True)
            .rename("proportion")
            .reset_index()
        )
        sns.barplot(data=proportions, x=feature, y="proportion", hue="Prediction", ax=ax)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{feature}: correct vs incorrect (overlap = {overlap:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
