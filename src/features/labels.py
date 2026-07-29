"""Label generation for supervised learning."""

from __future__ import annotations

import pandas as pd


def forward_return_column(horizon_days: int) -> str:
    """Return the standard forward-return column name for a horizon."""
    return f"forward_return_{horizon_days}d"


def compute_forward_returns(
    df: pd.DataFrame,
    horizon_days: int = 5,
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute forward returns without assigning class labels.

    Forward return at time t uses close[t+horizon] / close[t] - 1 so features
    at t only pair with outcomes realized after t.

    Args:
        df: Feature DataFrame with close prices.
        horizon_days: Forward return horizon in trading days.
        close_col: Name of close price column.

    Returns:
        DataFrame with a ``forward_return_{horizon}d`` column added.
    """
    labeled = df.copy()
    col = forward_return_column(horizon_days)
    labeled[col] = labeled[close_col].shift(-horizon_days) / labeled[close_col] - 1
    return labeled


def compute_relative_forward_returns(
    df: pd.DataFrame,
    horizon_days: int = 5,
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute cross-sectional relative (median-demeaned) forward returns.

    For each trading date, subtracts the within-date median absolute forward
    return from every stock's own forward return. This isolates the part of
    the return that separates winners from losers within a rebalance period
    (the market/day-common component cancels in a dollar-neutral long-short
    book anyway), which changes the *regression loss geometry* without
    changing the horizon.

    Requires the pooled panel (multiple symbols per date) already carrying the
    absolute forward-return column for ``horizon_days``; computes it via
    ``compute_forward_returns`` first if missing.

    Args:
        df: Pooled panel with date, symbol, and close columns.
        horizon_days: Horizon used to select/compute the forward-return column.
        close_col: Name of close price column.

    Returns:
        DataFrame with a ``forward_return_{horizon}d_rel`` column added.
    """
    ret_col = forward_return_column(horizon_days)
    frame = df if ret_col in df.columns else compute_forward_returns(
        df, horizon_days=horizon_days, close_col=close_col
    )
    rel_col = f"{ret_col}_rel"
    labeled = frame.copy()
    medians = labeled.groupby("date")[ret_col].transform("median")
    labeled[rel_col] = labeled[ret_col] - medians
    return labeled


def assign_cross_sectional_labels(
    df: pd.DataFrame,
    horizon_days: int = 5,
    positive_quantile: float = 0.20,
    min_names: int = 10,
) -> pd.DataFrame:
    """Assign binary labels from within-date forward-return ranks.

    For each trading date, stocks in the top ``positive_quantile`` of forward
    returns receive label 1; the remainder receive label 0. Dates with fewer
    than ``min_names`` valid returns are left unlabeled (NaN).

    Args:
        df: Pooled panel with date, symbol, and forward-return columns.
        horizon_days: Horizon used to select the forward-return column.
        positive_quantile: Fraction of names labeled positive each day (top).
        min_names: Minimum cross-section size required to assign labels.

    Returns:
        DataFrame with ``label`` column added.
    """
    if not 0.0 < positive_quantile < 1.0:
        raise ValueError("positive_quantile must be strictly between 0 and 1")
    if min_names < 2:
        raise ValueError("min_names must be at least 2")

    labeled = df.copy()
    ret_col = forward_return_column(horizon_days)
    if ret_col not in labeled.columns:
        raise ValueError(f"Missing forward-return column: {ret_col}")

    label_values = pd.Series(pd.NA, index=labeled.index, dtype="Float64")
    for _, group in labeled.groupby("date", sort=False):
        valid_idx = group.index[group[ret_col].notna()]
        n = len(valid_idx)
        if n < min_names:
            continue
        ranks = group.loc[valid_idx, ret_col].rank(method="first", ascending=True)
        pct = ranks / n
        label_values.loc[valid_idx] = (pct > (1.0 - positive_quantile)).astype(float)

    labeled["label"] = label_values
    return labeled


def generate_labels(
    df: pd.DataFrame,
    horizon_days: int = 5,
    threshold: float = 0.0,
    close_col: str = "close",
) -> pd.DataFrame:
    """Generate absolute binary labels from forward returns.

    Labels use future close prices shifted backward so features at time t
    only pair with outcomes computed from t+horizon.

    Args:
        df: Feature DataFrame with close prices.
        horizon_days: Forward return horizon in trading days.
        threshold: Minimum forward return for positive label.
        close_col: Name of close price column.

    Returns:
        DataFrame with forward_return and label columns added.
    """
    labeled = compute_forward_returns(df, horizon_days=horizon_days, close_col=close_col)
    ret_col = forward_return_column(horizon_days)
    labeled["label"] = (labeled[ret_col] > threshold).astype(float)
    return labeled


def drop_unlabeled_rows(df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """Remove rows without valid labels or features.

    Args:
        df: Labeled feature DataFrame.
        horizon_days: Horizon used for the forward-return column name.

    Returns:
        DataFrame with NaN label/feature rows removed.
    """
    ret_col = forward_return_column(horizon_days)
    exclude = {
        "symbol",
        "date",
        "label",
        ret_col,
        "forward_return_5d",
        "predicted_rank",
        "open",
        "high",
        "low",
        "volume",
        "close",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    subset = ["label", *feature_cols]
    return df.dropna(subset=subset).reset_index(drop=True)
