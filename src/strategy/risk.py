"""Risk management rules for portfolio construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.signals import assign_long_short_ranks, rank_signals_by_probability


@dataclass
class RiskLimits:
    """Configurable risk management limits."""

    max_positions: int = 10
    max_weight_per_symbol: float = 0.10
    entry_threshold: float = 0.55
    long_positions: int = 10
    short_positions: int = 10
    long_gross_exposure: float = 0.50
    short_gross_exposure: float = 0.50
    long_entry_threshold: float = 0.60
    short_entry_threshold: float = 0.40


def apply_risk_limits(
    df: pd.DataFrame,
    limits: RiskLimits,
    proba_col: str = "probability",
) -> pd.DataFrame:
    """Filter signals to respect position and probability limits.

    Args:
        df: DataFrame with date, symbol, probability, and signal.
        limits: Risk limit configuration.
        proba_col: Name of probability column.

    Returns:
        DataFrame with 'selected' column indicating approved positions.
    """
    ranked = rank_signals_by_probability(df, proba_col=proba_col)
    ranked["selected"] = (
        (ranked["signal"] == 1)
        & (ranked[proba_col] >= limits.entry_threshold)
        & (ranked["rank"] <= limits.max_positions)
    ).astype(int)
    return ranked


def compute_position_weights(
    df: pd.DataFrame,
    max_weight_per_symbol: float = 0.10,
) -> pd.DataFrame:
    """Assign equal weights to selected positions per rebalance date.

    Args:
        df: DataFrame with 'selected' column.
        max_weight_per_symbol: Maximum weight cap per symbol.

    Returns:
        DataFrame with 'weight' column.
    """
    result = df.copy()
    result["weight"] = 0.0

    for date, group in result.groupby("date"):
        selected = group[group["selected"] == 1]
        if selected.empty:
            continue
        equal_weight = min(1.0 / len(selected), max_weight_per_symbol)
        result.loc[selected.index, "weight"] = equal_weight

    return result


def apply_long_short_limits(
    df: pd.DataFrame,
    limits: RiskLimits,
    proba_col: str = "probability",
) -> pd.DataFrame:
    """Select a pure cross-sectional long/short book from probability ranks.

    Every rebalance date, the top ``long_positions`` symbols by probability are
    assigned to the long side and the bottom ``short_positions`` symbols are
    assigned to the short side, regardless of the absolute probability level
    (no confidence threshold gating).

    Args:
        df: DataFrame with date, symbol, and probability.
        limits: Risk limit configuration (long_positions, short_positions).
        proba_col: Name of probability column.

    Returns:
        DataFrame with 'long_rank', 'short_rank', 'side' ('long'/'short'/'flat'),
        and 'selected' (1 if side != 'flat') columns.
    """
    ranked = assign_long_short_ranks(df, proba_col=proba_col)

    is_long = ranked["long_rank"] <= limits.long_positions
    is_short = (~is_long) & (ranked["short_rank"] <= limits.short_positions)

    ranked["side"] = "flat"
    ranked.loc[is_long, "side"] = "long"
    ranked.loc[is_short, "side"] = "short"
    ranked["selected"] = (ranked["side"] != "flat").astype(int)
    return ranked


def apply_long_short_confidence_limits(
    df: pd.DataFrame,
    limits: RiskLimits,
    proba_col: str = "probability",
) -> pd.DataFrame:
    """Select a confidence-gated long/short book with position caps.

    Long candidates must have ``probability >= long_entry_threshold``; among
    those, the top ``long_positions`` by probability are selected. Short
    candidates must have ``probability <= short_entry_threshold``; among those,
    the bottom ``short_positions`` (lowest probability) are selected. Names
    that fail the gate are left flat, so the book can shrink or empty.

    Args:
        df: DataFrame with date, symbol, and probability.
        limits: Risk limits including confidence thresholds and position caps.
        proba_col: Name of probability column.

    Returns:
        DataFrame with ranks, side, and selected columns.
    """
    ranked = assign_long_short_ranks(df, proba_col=proba_col)

    long_eligible = ranked[proba_col] >= limits.long_entry_threshold
    short_eligible = ranked[proba_col] <= limits.short_entry_threshold

    # Re-rank within the confidence-eligible universe so caps apply only to
    # names that cleared the absolute probability gate.
    ranked["conf_long_rank"] = np.nan
    ranked["conf_short_rank"] = np.nan
    ranked.loc[long_eligible, "conf_long_rank"] = (
        ranked.loc[long_eligible]
        .groupby("date")[proba_col]
        .rank(ascending=False, method="first")
    )
    ranked.loc[short_eligible, "conf_short_rank"] = (
        ranked.loc[short_eligible]
        .groupby("date")[proba_col]
        .rank(ascending=True, method="first")
    )

    is_long = long_eligible & (ranked["conf_long_rank"] <= limits.long_positions)
    is_short = (
        (~is_long)
        & short_eligible
        & (ranked["conf_short_rank"] <= limits.short_positions)
    )

    ranked["side"] = "flat"
    ranked.loc[is_long, "side"] = "long"
    ranked.loc[is_short, "side"] = "short"
    ranked["selected"] = (ranked["side"] != "flat").astype(int)
    return ranked


def compute_long_short_weights(
    df: pd.DataFrame,
    limits: RiskLimits,
) -> pd.DataFrame:
    """Assign signed equal weights to a long/short book per rebalance date.

    Each date's long leg is equal-weighted to sum to ``long_gross_exposure``
    (positive weights) and the short leg is equal-weighted to sum to
    ``-short_gross_exposure`` (negative weights), with each leg's per-symbol
    weight capped in magnitude by ``max_weight_per_symbol``.

    Args:
        df: DataFrame with 'date' and 'side' columns.
        limits: Risk limit configuration.

    Returns:
        DataFrame with signed 'weight' column.
    """
    result = df.copy()
    result["weight"] = 0.0

    for _date, group in result.groupby("date"):
        longs = group[group["side"] == "long"]
        shorts = group[group["side"] == "short"]

        if not longs.empty:
            long_weight = min(
                limits.long_gross_exposure / len(longs),
                limits.max_weight_per_symbol,
            )
            result.loc[longs.index, "weight"] = long_weight

        if not shorts.empty:
            short_weight = min(
                limits.short_gross_exposure / len(shorts),
                limits.max_weight_per_symbol,
            )
            result.loc[shorts.index, "weight"] = -short_weight

    return result
