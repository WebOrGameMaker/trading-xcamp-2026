"""Tests for strategy signals, risk management, and rebalance scheduling."""

import pandas as pd

from src.strategy.portfolio import build_portfolio_signals
from src.strategy.rebalance import weekly_rebalance_dates
from src.strategy.risk import (
    RiskLimits,
    apply_long_short_confidence_limits,
    apply_long_short_limits,
    apply_risk_limits,
    compute_long_short_weights,
)
from src.strategy.signals import assign_long_short_ranks, probability_to_signals
from src.utils.config import StrategyConfig


def _sample_predictions() -> pd.DataFrame:
    """Create sample prediction DataFrame."""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    rows = []
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    probs = [0.7, 0.65, 0.6, 0.55, 0.45]
    for date in dates:
        for sym, prob in zip(symbols, probs, strict=True):
            rows.append({"date": date, "symbol": sym, "probability": prob, "close": 100.0})
    return pd.DataFrame(rows)


def test_probability_to_signals() -> None:
    """Signals generated above entry threshold."""
    df = _sample_predictions()
    signals = probability_to_signals(df, entry_threshold=0.55)
    assert signals.loc[signals["probability"] >= 0.55, "signal"].eq(1).all()
    assert signals.loc[signals["probability"] < 0.55, "signal"].eq(0).all()


def test_max_positions_limit() -> None:
    """Risk limits cap number of selected positions (legacy long-only mode)."""
    df = _sample_predictions()
    signals = probability_to_signals(df, entry_threshold=0.55)
    filtered = apply_risk_limits(signals, RiskLimits(max_positions=3, entry_threshold=0.55))
    for date in filtered["date"].unique():
        selected = filtered[(filtered["date"] == date) & (filtered["selected"] == 1)]
        assert len(selected) <= 3


def test_portfolio_weights_sum() -> None:
    """Selected position weights do not exceed 100% (legacy long-only mode)."""
    config = StrategyConfig(
        mode="long_only", max_positions=3, max_weight_per_symbol=0.5, entry_threshold=0.55
    )
    df = _sample_predictions()
    portfolio = build_portfolio_signals(df, config)
    for date in portfolio["date"].unique():
        day = portfolio[(portfolio["date"] == date) & (portfolio["selected"] == 1)]
        if not day.empty:
            assert day["weight"].sum() <= 1.0 + 1e-6


def test_assign_long_short_ranks() -> None:
    """Long rank favors highest probability, short rank favors lowest."""
    df = _sample_predictions()
    ranked = assign_long_short_ranks(df)
    for date in ranked["date"].unique():
        day = ranked[ranked["date"] == date]
        best_long = day.loc[day["long_rank"] == 1, "symbol"].iloc[0]
        best_short = day.loc[day["short_rank"] == 1, "symbol"].iloc[0]
        assert best_long == "AAPL"  # highest probability (0.70)
        assert best_short == "META"  # lowest probability (0.45)


def test_apply_long_short_limits_top_bottom() -> None:
    """Top-N by probability go long, bottom-N go short, rest stay flat."""
    df = _sample_predictions()
    limits = RiskLimits(long_positions=2, short_positions=2)
    selected = apply_long_short_limits(df, limits)
    for date in selected["date"].unique():
        day = selected[selected["date"] == date]
        longs = set(day.loc[day["side"] == "long", "symbol"])
        shorts = set(day.loc[day["side"] == "short", "symbol"])
        flats = set(day.loc[day["side"] == "flat", "symbol"])
        assert longs == {"AAPL", "MSFT"}
        assert shorts == {"META", "AMZN"}
        assert flats == {"GOOG"}
        assert longs.isdisjoint(shorts)


def test_compute_long_short_weights_gross_exposure() -> None:
    """Long leg sums to +gross exposure, short leg sums to -gross exposure."""
    df = _sample_predictions()
    limits = RiskLimits(
        long_positions=2,
        short_positions=2,
        long_gross_exposure=0.5,
        short_gross_exposure=0.5,
        max_weight_per_symbol=0.5,
    )
    selected = apply_long_short_limits(df, limits)
    weighted = compute_long_short_weights(selected, limits)
    for date in weighted["date"].unique():
        day = weighted[weighted["date"] == date]
        long_weight = day.loc[day["side"] == "long", "weight"].sum()
        short_weight = day.loc[day["side"] == "short", "weight"].sum()
        flat_weight = day.loc[day["side"] == "flat", "weight"].sum()
        assert long_weight == 0.5
        assert short_weight == -0.5
        assert flat_weight == 0.0
        assert day["weight"].abs().max() <= limits.max_weight_per_symbol + 1e-9


def test_build_portfolio_signals_long_short_mode() -> None:
    """End-to-end long/short portfolio is market-neutral with capped weights."""
    config = StrategyConfig(
        mode="long_short",
        long_positions=2,
        short_positions=2,
        long_gross_exposure=0.5,
        short_gross_exposure=0.5,
        max_weight_per_symbol=0.5,
    )
    df = _sample_predictions()
    portfolio = build_portfolio_signals(df, config)
    for date in portfolio["date"].unique():
        day = portfolio[portfolio["date"] == date]
        assert abs(day["weight"].sum()) < 1e-9  # net exposure ~ 0 (market-neutral)
        selected = day[day["selected"] == 1]
        assert len(selected) == 4  # 2 long + 2 short out of 5 symbols


def test_apply_long_short_confidence_limits_filters_low_confidence() -> None:
    """Confidence gate drops names that fail absolute probability thresholds."""
    df = _sample_predictions()
    limits = RiskLimits(
        long_positions=3,
        short_positions=3,
        long_entry_threshold=0.65,
        short_entry_threshold=0.50,
    )
    selected = apply_long_short_confidence_limits(df, limits)
    for date in selected["date"].unique():
        day = selected[selected["date"] == date]
        longs = set(day.loc[day["side"] == "long", "symbol"])
        shorts = set(day.loc[day["side"] == "short", "symbol"])
        # Only AAPL (0.70) and MSFT (0.65) clear long_entry_threshold=0.65.
        assert longs == {"AAPL", "MSFT"}
        # Only META (0.45) clears short_entry_threshold=0.50.
        assert shorts == {"META"}


def test_apply_long_short_confidence_limits_empty_when_nothing_passes() -> None:
    """Extreme thresholds leave the entire book flat."""
    df = _sample_predictions()
    limits = RiskLimits(
        long_positions=2,
        short_positions=2,
        long_entry_threshold=0.95,
        short_entry_threshold=0.05,
    )
    selected = apply_long_short_confidence_limits(df, limits)
    assert (selected["side"] == "flat").all()
    assert (selected["selected"] == 0).all()


def test_build_portfolio_signals_confidence_mode() -> None:
    """Confidence mode builds a possibly smaller long/short book."""
    config = StrategyConfig(
        mode="long_short_confidence",
        long_positions=2,
        short_positions=2,
        long_gross_exposure=0.5,
        short_gross_exposure=0.5,
        max_weight_per_symbol=0.5,
        long_entry_threshold=0.65,
        short_entry_threshold=0.50,
    )
    df = _sample_predictions()
    portfolio = build_portfolio_signals(df, config)
    for date in portfolio["date"].unique():
        day = portfolio[portfolio["date"] == date]
        selected = day[day["selected"] == 1]
        assert len(selected) == 3  # 2 long + 1 short
        assert set(selected.loc[selected["side"] == "long", "symbol"]) == {"AAPL", "MSFT"}
        assert set(selected.loc[selected["side"] == "short", "symbol"]) == {"META"}


def test_weekly_rebalance_dates() -> None:
    """Weekly rebalance picks one trading date per calendar week."""
    dates = pd.date_range("2024-01-01", "2024-01-19", freq="B")  # 3 full business weeks
    rebalance_dates = weekly_rebalance_dates(dates)
    assert len(rebalance_dates) == 3
    # Each rebalance date should be a date that actually exists in the input.
    assert set(rebalance_dates).issubset(set(dates))
    # Rebalance dates should be strictly increasing.
    assert list(rebalance_dates) == sorted(rebalance_dates)
