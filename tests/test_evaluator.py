"""Tests for classification evaluation metrics, including calibration."""

from __future__ import annotations

import numpy as np

from src.models.evaluator import compute_ece, evaluate_classifier, evaluate_regressor


def test_evaluate_classifier_reports_brier_score() -> None:
    """A confident, correct classifier has a low Brier score."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_proba = np.array([0.05, 0.1, 0.9, 0.95, 0.05, 0.85, 0.1, 0.9])

    metrics = evaluate_classifier(y_true, y_pred, y_proba)

    assert metrics.brier_score < 0.05
    assert metrics.accuracy == 1.0
    assert metrics.pr_auc > 0.9
    assert metrics.ece < 0.15


def test_evaluate_classifier_reports_pr_auc() -> None:
    """PR-AUC is high when positives are ranked above negatives."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.9, 0.85, 0.95])

    metrics = evaluate_classifier(y_true, y_pred, y_proba)

    assert metrics.pr_auc > 0.9
    assert metrics.roc_auc > 0.9


def test_evaluate_classifier_brier_score_penalizes_overconfidence() -> None:
    """A classifier that is confidently wrong on every sample scores near the worst possible Brier score."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    confidently_wrong = np.array([0.95, 0.9, 0.92, 0.97, 0.05, 0.1, 0.08, 0.03])

    metrics = evaluate_classifier(y_true, y_pred, confidently_wrong)

    assert metrics.brier_score > 0.7
    assert metrics.accuracy == 0.0
    assert metrics.ece > 0.5


def test_calibration_bins_track_predicted_vs_actual_frequency() -> None:
    """Calibration bins summarize mean predicted probability vs. realized outcome."""
    rng = np.random.default_rng(42)
    y_proba = rng.uniform(0, 1, size=500)
    # Perfectly calibrated by construction: P(y=1) == predicted probability.
    y_true = (rng.uniform(0, 1, size=500) < y_proba).astype(int)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = evaluate_classifier(y_true, y_pred, y_proba)

    assert len(metrics.calibration_bins) > 1
    assert len(metrics.reliability_bins) > 1
    for bin_stats in metrics.calibration_bins:
        assert {"mean_predicted", "fraction_positive", "count"} <= bin_stats.keys()
        # With 500 samples and a genuinely calibrated generator, bins should
        # not be wildly off from the diagonal.
        assert abs(bin_stats["mean_predicted"] - bin_stats["fraction_positive"]) < 0.35
    assert metrics.ece < 0.1


def test_compute_ece_is_zero_for_perfect_calibration() -> None:
    """ECE is near zero when predicted confidence matches realized frequency."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    y_proba = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    ece = compute_ece(y_true, y_proba, n_bins=2)

    assert ece == 0.0


def test_calibration_bins_empty_when_too_few_samples() -> None:
    """Degenerate inputs degrade gracefully to an empty calibration report."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_proba = np.array([0.5, 0.5])

    metrics = evaluate_classifier(y_true, y_pred, y_proba)

    assert metrics.calibration_bins == []
    # Equal-width bins still form a single occupied bin for identical scores.
    assert metrics.reliability_bins
    assert metrics.ece >= 0.0


def test_evaluate_classifier_single_class_skips_probability_metrics() -> None:
    """AUC, Brier, and calibration require both classes to be meaningful."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.15, 0.05])

    metrics = evaluate_classifier(y_true, y_pred, y_proba)

    assert metrics.roc_auc == 0.0
    assert metrics.brier_score == 0.0
    assert metrics.ece == 0.0
    assert metrics.calibration_bins == []
    assert metrics.reliability_bins == []


def test_evaluate_regressor_perfect_fit() -> None:
    """Identical predictions yield zero error and R² = 1."""
    y_true = np.array([0.01, -0.02, 0.03, 0.0])
    metrics = evaluate_regressor(y_true, y_true.copy())
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.r2 == 1.0
    assert metrics.support == 4


def test_evaluate_regressor_reports_positive_rmse() -> None:
    """A constant wrong prediction has positive RMSE and non-perfect R²."""
    y_true = np.array([0.01, -0.02, 0.03, 0.0, 0.05])
    y_pred = np.zeros_like(y_true)
    metrics = evaluate_regressor(y_true, y_pred)
    assert metrics.rmse > 0.0
    assert metrics.mae > 0.0
    assert metrics.r2 < 1.0
    assert metrics.support == 5
