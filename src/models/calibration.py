"""Probability calibration (Platt scaling and isotonic regression).

Calibrators are always fit on validation-set raw scores after the base
classifier is trained on the training set. Test data is never used for
fitting or method selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils.logging import get_logger

logger = get_logger(__name__)

PROBABILITY_COLUMNS = ("probability", "probability_platt", "probability_isotonic")


@dataclass
class FittedCalibrators:
    """Fitted Platt (sigmoid) and isotonic calibrators."""

    platt: LogisticRegression
    isotonic: IsotonicRegression

    def transform(self, raw_proba: np.ndarray) -> dict[str, np.ndarray]:
        """Apply both calibrators to raw positive-class probabilities.

        Args:
            raw_proba: Uncalibrated probabilities in [0, 1].

        Returns:
            Mapping with keys ``platt`` and ``isotonic``.
        """
        return {
            "platt": apply_calibrator(self.platt, raw_proba, method="platt"),
            "isotonic": apply_calibrator(self.isotonic, raw_proba, method="isotonic"),
        }


def fit_calibrators(y_val: np.ndarray, p_val: np.ndarray) -> FittedCalibrators:
    """Fit Platt and isotonic calibrators on validation scores only.

    Args:
        y_val: Validation labels (0/1).
        p_val: Raw validation probabilities from the base model.

    Returns:
        FittedCalibrators with both methods ready to apply.

    Raises:
        ValueError: If validation labels lack both classes.
    """
    y = np.asarray(y_val).astype(int).ravel()
    p = np.asarray(p_val, dtype=float).ravel()
    if len(np.unique(y)) < 2:
        raise ValueError("Validation labels must contain both classes to fit calibrators")

    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(p.reshape(-1, 1), y)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(p, y)

    logger.info(
        "Fitted Platt and isotonic calibrators on %d validation samples "
        "(pos_rate=%.3f, mean_raw_p=%.3f)",
        len(y),
        float(y.mean()),
        float(p.mean()),
    )
    return FittedCalibrators(platt=platt, isotonic=isotonic)


def apply_calibrator(
    calibrator: Any,
    raw_proba: np.ndarray,
    method: str,
) -> np.ndarray:
    """Apply a fitted calibrator to raw probabilities.

    Args:
        calibrator: Fitted LogisticRegression (Platt) or IsotonicRegression.
        raw_proba: Uncalibrated probabilities.
        method: ``platt`` or ``isotonic``.

    Returns:
        Calibrated probabilities clipped to [0, 1].
    """
    p = np.asarray(raw_proba, dtype=float).ravel()
    if method == "platt":
        calibrated = calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        calibrated = calibrator.predict(p)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


def save_calibrators(calibrators: FittedCalibrators, path: str | Path) -> Path:
    """Persist fitted calibrators to disk.

    Args:
        calibrators: Fitted Platt + isotonic pair.
        path: Destination joblib path.

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"platt": calibrators.platt, "isotonic": calibrators.isotonic}, path)
    return path


def load_calibrators(path: str | Path) -> FittedCalibrators:
    """Load fitted calibrators from disk.

    Args:
        path: Joblib path written by :func:`save_calibrators`.

    Returns:
        FittedCalibrators instance.
    """
    payload = joblib.load(path)
    return FittedCalibrators(platt=payload["platt"], isotonic=payload["isotonic"])
