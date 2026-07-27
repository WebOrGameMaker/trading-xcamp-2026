"""Cross-sectional ranking and information-coefficient metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def information_coefficient(
    predictions: pd.DataFrame,
    score_col: str = "probability",
    return_col: str | None = None,
) -> dict[str, float]:
    """Compute Spearman IC between predicted scores and forward returns.

    Args:
        predictions: Panel with scores, returns, and dates.
        score_col: Predicted score column.
        return_col: Forward-return column; inferred if None.

    Returns:
        Dict with overall IC and mean daily IC.
    """
    ret_col = return_col or _infer_return_column(predictions)
    frame = predictions[[score_col, ret_col, "date"]].dropna()
    if len(frame) < 3:
        return {"ic_overall": 0.0, "ic_mean_daily": 0.0, "ic_std_daily": 0.0}

    overall = float(frame[score_col].corr(frame[ret_col], method="spearman") or 0.0)
    daily_vals: list[float] = []
    for _, group in frame.groupby("date", sort=False):
        if len(group) < 3:
            continue
        corr = group[score_col].corr(group[ret_col], method="spearman")
        if pd.notna(corr):
            daily_vals.append(float(corr))
    daily = pd.Series(daily_vals, dtype=float)
    return {
        "ic_overall": overall,
        "ic_mean_daily": float(daily.mean()) if len(daily) else 0.0,
        "ic_std_daily": float(daily.std()) if len(daily) else 0.0,
    }


def top_decile_hit_rate(
    predictions: pd.DataFrame,
    score_col: str = "probability",
    label_col: str = "label",
    top_frac: float = 0.10,
) -> float:
    """Fraction of top-``top_frac`` predicted names that have a positive label.

    Args:
        predictions: Panel with scores and labels.
        score_col: Predicted score column.
        label_col: Binary label column.
        top_frac: Within-date top fraction of predictions to evaluate.

    Returns:
        Hit rate in [0, 1].
    """
    hits: list[float] = []
    for _, group in predictions.groupby("date", sort=False):
        if group[score_col].isna().all() or group[label_col].isna().all():
            continue
        n = len(group)
        k = max(1, int(np.ceil(n * top_frac)))
        top = group.nlargest(k, score_col)
        hits.append(float(top[label_col].mean()))
    return float(np.mean(hits)) if hits else 0.0


def mean_return_by_prediction_decile(
    predictions: pd.DataFrame,
    score_col: str = "probability",
    return_col: str | None = None,
    n_deciles: int = 10,
) -> list[dict[str, float]]:
    """Average forward return within prediction-score deciles.

    Args:
        predictions: Panel with scores and forward returns.
        score_col: Predicted score column.
        return_col: Forward-return column; inferred if None.
        n_deciles: Number of equal-count score buckets.

    Returns:
        List of {decile, mean_forward_return, count} ordered low→high score.
    """
    ret_col = return_col or _infer_return_column(predictions)
    frame = predictions[[score_col, ret_col]].dropna().copy()
    if frame.empty:
        return []

    try:
        frame["decile"] = pd.qcut(
            frame[score_col],
            q=n_deciles,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return []

    rows: list[dict[str, float]] = []
    for decile, group in frame.groupby("decile", observed=True):
        rows.append({
            "decile": int(decile) + 1,
            "mean_forward_return": float(group[ret_col].mean()),
            "count": int(len(group)),
        })
    return rows


def evaluate_cross_sectional(
    predictions: pd.DataFrame,
    score_col: str = "probability",
    label_col: str = "label",
) -> dict[str, Any]:
    """Compute the full cross-sectional evaluation bundle.

    Args:
        predictions: Prediction panel for one split.
        score_col: Predicted score column.
        label_col: Binary label column.

    Returns:
        Dict of IC, hit-rate, and decile-return metrics.
    """
    ret_col = _infer_return_column(predictions)
    ic = information_coefficient(predictions, score_col=score_col, return_col=ret_col)
    return {
        **ic,
        "top_decile_hit_rate": top_decile_hit_rate(
            predictions,
            score_col=score_col,
            label_col=label_col,
        ),
        "mean_return_by_prediction_decile": mean_return_by_prediction_decile(
            predictions,
            score_col=score_col,
            return_col=ret_col,
        ),
    }


def _infer_return_column(df: pd.DataFrame) -> str:
    """Pick the forward-return column from a prediction frame."""
    candidates = [c for c in df.columns if c.startswith("forward_return_")]
    if not candidates:
        raise ValueError("No forward_return_* column found in predictions")
    return candidates[0]
