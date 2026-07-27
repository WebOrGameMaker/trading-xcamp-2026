"""Trading signal generation from model predictions."""

from __future__ import annotations

import pandas as pd


def probability_to_signals(
    df: pd.DataFrame,
    entry_threshold: float = 0.55,
    proba_col: str = "probability",
) -> pd.DataFrame:
    """Convert model probabilities to binary long signals.

    Args:
        df: DataFrame with probability column.
        entry_threshold: Minimum probability to enter long.
        proba_col: Name of probability column.

    Returns:
        DataFrame with 'signal' column (1 = long, 0 = flat).
    """
    result = df.copy()
    result["signal"] = (result[proba_col] >= entry_threshold).astype(int)
    return result


def rank_signals_by_probability(df: pd.DataFrame, proba_col: str = "probability") -> pd.DataFrame:
    """Rank symbols by probability within each date.

    Args:
        df: DataFrame with date, symbol, and probability.
        proba_col: Name of probability column.

    Returns:
        DataFrame with 'rank' column (1 = highest probability).
    """
    result = df.copy()
    result["rank"] = result.groupby("date")[proba_col].rank(ascending=False, method="first")
    return result


def assign_long_short_ranks(df: pd.DataFrame, proba_col: str = "probability") -> pd.DataFrame:
    """Rank symbols for a cross-sectional long/short strategy within each date.

    Args:
        df: DataFrame with date, symbol, and probability.
        proba_col: Name of probability column.

    Returns:
        DataFrame with 'long_rank' (1 = highest probability, best long candidate)
        and 'short_rank' (1 = lowest probability, best short candidate) columns.
    """
    result = df.copy()
    result["long_rank"] = result.groupby("date")[proba_col].rank(ascending=False, method="first")
    result["short_rank"] = result.groupby("date")[proba_col].rank(ascending=True, method="first")
    return result
