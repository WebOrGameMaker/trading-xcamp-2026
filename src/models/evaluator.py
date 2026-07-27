"""Model evaluation metrics and reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging import get_logger
from src.utils.paths import LOG_DIR

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    """Classification evaluation results."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]
    support: int
    brier_score: float
    calibration_bins: list[dict[str, float]]
    reliability_bins: list[dict[str, float]]
    ece: float


def _bin_stats_from_groups(grouped: pd.DataFrame) -> list[dict[str, float]]:
    """Convert aggregated bin statistics into serializable dicts."""
    return [
        {
            "mean_predicted": float(row.mean_predicted),
            "fraction_positive": float(row.fraction_positive),
            "count": int(row.count),
        }
        for row in grouped.itertuples()
    ]


def _compute_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Bin predicted probabilities with quantile cuts (diagnostic reliability).

    A well-calibrated model has ``fraction_positive`` roughly equal to
    ``mean_predicted`` in every bin. Systematic deviation (e.g. predicted
    0.7 but only 50% actually positive) means the probabilities cannot be
    trusted for confidence-gating as-is.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities for the positive class.
        n_bins: Number of quantile bins to group predictions into.

    Returns:
        List of per-bin dicts with mean_predicted, fraction_positive, and
        count, ordered from lowest to highest predicted probability. Empty
        if there are too few samples/unique values to form bins.
    """
    frame = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    try:
        frame["bin"] = pd.qcut(frame["y_proba"], q=n_bins, duplicates="drop")
    except ValueError:
        return []

    grouped = frame.groupby("bin", observed=True).agg(
        mean_predicted=("y_proba", "mean"),
        fraction_positive=("y_true", "mean"),
        count=("y_true", "size"),
    )
    return _bin_stats_from_groups(grouped)


def compute_reliability_bins(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Equal-width probability bins for reliability diagrams and ECE.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities for the positive class.
        n_bins: Number of equal-width bins on [0, 1].

    Returns:
        Non-empty bins only, each with mean_predicted, fraction_positive, count.
    """
    if len(y_true) == 0:
        return []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Rightmost edge is inclusive so p=1.0 lands in the last bin.
    bin_ids = np.digitize(y_proba, edges[1:-1], right=False)
    frame = pd.DataFrame({"y_true": y_true, "y_proba": y_proba, "bin": bin_ids})
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            mean_predicted=("y_proba", "mean"),
            fraction_positive=("y_true", "mean"),
            count=("y_true", "size"),
        )
        .sort_index()
    )
    return _bin_stats_from_groups(grouped)


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    reliability_bins: list[dict[str, float]] | None = None,
) -> float:
    """Expected Calibration Error from equal-width reliability bins.

    ECE = sum_b (n_b / N) * |acc_b - conf_b|.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities for the positive class.
        n_bins: Number of equal-width bins when ``reliability_bins`` is None.
        reliability_bins: Optional precomputed equal-width bins.

    Returns:
        ECE in [0, 1]. Returns 0.0 when bins cannot be formed.
    """
    bins = reliability_bins
    if bins is None:
        bins = compute_reliability_bins(y_true, y_proba, n_bins=n_bins)
    total = sum(int(b["count"]) for b in bins)
    if total == 0:
        return 0.0
    return float(
        sum(
            (int(b["count"]) / total)
            * abs(float(b["fraction_positive"]) - float(b["mean_predicted"]))
            for b in bins
        )
    )


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute classification metrics for binary predictions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        y_proba: Predicted probabilities for positive class.

    Returns:
        ClassificationMetrics dataclass.
    """
    roc_auc = 0.0
    pr_auc = 0.0
    brier = 0.0
    ece = 0.0
    calibration_bins: list[dict[str, float]] = []
    reliability_bins: list[dict[str, float]] = []
    if y_proba is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))
        brier = float(brier_score_loss(y_true, y_proba))
        calibration_bins = _compute_calibration(y_true, y_proba)
        reliability_bins = compute_reliability_bins(y_true, y_proba)
        ece = compute_ece(y_true, y_proba, reliability_bins=reliability_bins)

    cm = confusion_matrix(y_true, y_pred).tolist()
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm,
        support=len(y_true),
        brier_score=brier,
        calibration_bins=calibration_bins,
        reliability_bins=reliability_bins,
        ece=ece,
    )


def save_evaluation_report(
    metrics: ClassificationMetrics,
    split_name: str,
    run_id: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save evaluation metrics to logs directory as JSON.

    Args:
        metrics: Classification metrics.
        split_name: Dataset split name (train/val/test).
        run_id: Unique run identifier.
        extra: Optional additional fields.

    Returns:
        Path to saved report file.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report = asdict(metrics)
    report["split"] = split_name
    report["run_id"] = run_id
    if extra:
        report.update(extra)

    path = LOG_DIR / f"eval_{run_id}_{split_name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info(
        "%s metrics — accuracy: %.3f, f1: %.3f, roc_auc: %.3f, pr_auc: %.3f, "
        "brier: %.3f, ece: %.3f",
        split_name,
        metrics.accuracy,
        metrics.f1,
        metrics.roc_auc,
        metrics.pr_auc,
        metrics.brier_score,
        metrics.ece,
    )
    return path


def extract_feature_importance(pipeline: Any, feature_columns: list[str]) -> dict[str, float]:
    """Extract feature importance from a fitted pipeline if available.

    Args:
        pipeline: Fitted sklearn Pipeline with classifier step.
        feature_columns: Feature column names.

    Returns:
        Mapping of feature name to importance score.
    """
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        return {}

    importances: np.ndarray | None = None
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).ravel()

    if importances is None or len(importances) != len(feature_columns):
        return {}

    return dict(sorted(
        zip(feature_columns, importances.astype(float), strict=True),
        key=lambda x: x[1],
        reverse=True,
    ))


def compute_shap_importance(
    pipeline: Any,
    X: np.ndarray | pd.DataFrame,
    feature_columns: list[str],
    max_samples: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute mean |SHAP| feature importance for a tree classifier.

    Args:
        pipeline: Fitted sklearn Pipeline with scaler + classifier.
        X: Feature matrix (unscaled; pipeline scaler is applied inside).
        feature_columns: Feature names aligned with columns of X.
        max_samples: Cap on rows used for SHAP (speed).
        random_state: RNG seed for subsampling.

    Returns:
        Mapping of feature name to mean absolute SHAP value, sorted desc.
        Empty dict if SHAP is unavailable or computation fails.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed; skipping SHAP importance")
        return {}

    classifier = pipeline.named_steps.get("classifier")
    scaler = pipeline.named_steps.get("scaler")
    if classifier is None:
        return {}

    if isinstance(X, pd.DataFrame):
        matrix = X[feature_columns].to_numpy()
    else:
        matrix = np.asarray(X)

    if len(matrix) == 0:
        return {}

    rng = np.random.default_rng(random_state)
    if len(matrix) > max_samples:
        idx = rng.choice(len(matrix), size=max_samples, replace=False)
        matrix = matrix[idx]

    if scaler is not None:
        matrix = scaler.transform(matrix)

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(matrix)
    except Exception as exc:  # noqa: BLE001 — SHAP is best-effort
        logger.warning("SHAP computation failed: %s", exc)
        return {}

    if isinstance(shap_values, list):
        # Binary classifiers may return [neg, pos]; use positive class.
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values

    values = np.asarray(values)
    if values.ndim == 3:
        # Newer shap versions return (n_samples, n_features, n_classes) for
        # multi-output-capable explainers (e.g. RandomForestClassifier) instead
        # of a list of per-class arrays; select the positive class slice.
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]

    mean_abs = np.abs(values).mean(axis=0)
    if len(mean_abs) != len(feature_columns):
        return {}

    return dict(sorted(
        zip(feature_columns, mean_abs.astype(float), strict=True),
        key=lambda x: x[1],
        reverse=True,
    ))
