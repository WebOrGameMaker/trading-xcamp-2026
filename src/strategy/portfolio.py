"""Portfolio construction and rebalancing logic."""

from __future__ import annotations

import pandas as pd

from src.strategy.risk import (
    RiskLimits,
    apply_long_short_confidence_limits,
    apply_long_short_limits,
    apply_risk_limits,
    compute_long_short_weights,
    compute_position_weights,
)
from src.strategy.signals import probability_to_signals
from src.utils.config import StrategyConfig


def _resolve_proba_col(predictions: pd.DataFrame, config: StrategyConfig) -> str:
    """Return the configured probability column, falling back to raw if missing."""
    col = getattr(config, "probability_column", "probability") or "probability"
    if col not in predictions.columns:
        if col != "probability" and "probability" in predictions.columns:
            return "probability"
        raise KeyError(
            f"Probability column {col!r} not found in predictions. "
            f"Available: {list(predictions.columns)}"
        )
    return col


def _build_long_short_portfolio(
    predictions: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Build a weekly market-neutral long/short portfolio from predictions.

    Every rebalance date, ranks the full universe by predicted probability and
    goes long the top ``long_positions`` symbols and short the bottom
    ``short_positions`` symbols (pure rank, no confidence threshold).

    Args:
        predictions: DataFrame with date, symbol, probability columns.
        config: Strategy configuration.

    Returns:
        DataFrame with long_rank, short_rank, side, selected, and signed weight columns.
    """
    proba_col = _resolve_proba_col(predictions, config)
    limits = RiskLimits(
        long_positions=config.long_positions,
        short_positions=config.short_positions,
        max_weight_per_symbol=config.max_weight_per_symbol,
        long_gross_exposure=config.long_gross_exposure,
        short_gross_exposure=config.short_gross_exposure,
    )
    ranked = apply_long_short_limits(predictions, limits, proba_col=proba_col)
    portfolio = compute_long_short_weights(ranked, limits)
    return portfolio


def _build_long_short_confidence_portfolio(
    predictions: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Build a confidence-filtered long/short portfolio.

    Longs require probability >= long_entry_threshold (capped at long_positions);
    shorts require probability <= short_entry_threshold (capped at short_positions).

    Args:
        predictions: DataFrame with date, symbol, probability columns.
        config: Strategy configuration.

    Returns:
        DataFrame with side, selected, and signed weight columns.
    """
    proba_col = _resolve_proba_col(predictions, config)
    limits = RiskLimits(
        long_positions=config.long_positions,
        short_positions=config.short_positions,
        max_weight_per_symbol=config.max_weight_per_symbol,
        long_gross_exposure=config.long_gross_exposure,
        short_gross_exposure=config.short_gross_exposure,
        long_entry_threshold=config.long_entry_threshold,
        short_entry_threshold=config.short_entry_threshold,
    )
    ranked = apply_long_short_confidence_limits(predictions, limits, proba_col=proba_col)
    portfolio = compute_long_short_weights(ranked, limits)
    return portfolio


def _build_long_only_portfolio(
    predictions: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Build a legacy long-only, threshold-gated portfolio from predictions.

    Args:
        predictions: DataFrame with date, symbol, probability columns.
        config: Strategy configuration.

    Returns:
        DataFrame with signal, selected, and weight columns.
    """
    proba_col = _resolve_proba_col(predictions, config)
    limits = RiskLimits(
        max_positions=config.max_positions,
        max_weight_per_symbol=config.max_weight_per_symbol,
        entry_threshold=config.entry_threshold,
    )

    signals = probability_to_signals(
        predictions, entry_threshold=config.entry_threshold, proba_col=proba_col
    )
    filtered = apply_risk_limits(signals, limits, proba_col=proba_col)
    portfolio = compute_position_weights(
        filtered, max_weight_per_symbol=config.max_weight_per_symbol
    )
    return portfolio


def build_portfolio_signals(
    predictions: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Build risk-adjusted portfolio signals from model predictions.

    Dispatches to:
    - ``long_short``: pure cross-sectional top/bottom-N ranking (default)
    - ``long_short_confidence``: confidence-gated long/short with position caps
    - ``long_only``: legacy long-only threshold mode

    Args:
        predictions: DataFrame with date, symbol, probability columns.
        config: Strategy configuration.

    Returns:
        DataFrame with selected and (signed) weight columns.
    """
    mode = getattr(config, "mode", "long_short")
    if mode == "long_short":
        return _build_long_short_portfolio(predictions, config)
    if mode == "long_short_confidence":
        return _build_long_short_confidence_portfolio(predictions, config)
    if mode == "long_only":
        return _build_long_only_portfolio(predictions, config)
    raise ValueError(f"Unknown strategy mode: {mode}")


def get_latest_targets(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Get target positions for the most recent date in portfolio signals.

    Args:
        portfolio: Portfolio signals DataFrame.

    Returns:
        DataFrame of selected symbols with (signed) weights for latest date.
    """
    latest_date = portfolio["date"].max()
    latest = portfolio[(portfolio["date"] == latest_date) & (portfolio["selected"] == 1)]
    columns = ["symbol", "weight", "probability"]
    if "side" in latest.columns:
        columns.append("side")
    return latest[columns].copy()
