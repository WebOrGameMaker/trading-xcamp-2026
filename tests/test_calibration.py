"""Tests for Platt and isotonic probability calibration."""

from __future__ import annotations

import numpy as np

from src.models.calibration import apply_calibrator, fit_calibrators


def test_fit_calibrators_improves_overconfident_scores() -> None:
    """Calibrators pull systematically overconfident scores toward the base rate."""
    rng = np.random.default_rng(0)
    # Labels with ~20% positives; raw scores centered near 0.6 (overconfident).
    y = (rng.uniform(0, 1, size=2000) < 0.2).astype(int)
    raw = np.clip(0.55 + 0.15 * rng.normal(size=2000) + 0.2 * y, 0.01, 0.99)

    calibrators = fit_calibrators(y, raw)
    platt = apply_calibrator(calibrators.platt, raw, method="platt")
    isotonic = apply_calibrator(calibrators.isotonic, raw, method="isotonic")

    assert platt.mean() < raw.mean()
    assert isotonic.mean() < raw.mean()
    assert 0.0 <= platt.min() <= platt.max() <= 1.0
    assert 0.0 <= isotonic.min() <= isotonic.max() <= 1.0


def test_apply_calibrator_preserves_length() -> None:
    """Transformed probabilities keep the same length as the input."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    calibrators = fit_calibrators(y, p)
    out = calibrators.transform(p)
    assert len(out["platt"]) == len(p)
    assert len(out["isotonic"]) == len(p)
