"""Backtest performance metrics calculation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Trading backtest performance metrics."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    benchmark_return: float
    turnover: float = 0.0


def compute_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio from daily returns.

    Args:
        returns: Series of periodic returns.
        periods_per_year: Number of periods per year for annualization.

    Returns:
        Annualized Sharpe ratio, or 0.0 if std is zero.
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def compute_annualized_return(
    equity: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized return from an equity curve.

    Args:
        equity: Cumulative equity series.
        periods_per_year: Trading periods per year.

    Returns:
        Annualized return as a fraction.
    """
    if len(equity) < 2:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_periods = len(equity) - 1
    if n_periods <= 0 or equity.iloc[0] == 0:
        return 0.0
    return float((1.0 + total) ** (periods_per_year / n_periods) - 1.0)


def compute_max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve.

    Args:
        equity: Cumulative equity series.

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.15 = 15% drawdown).
    """
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max.replace(0, np.nan)
    return float(abs(drawdown.min()))


def compute_win_rate(trades: pd.DataFrame) -> float:
    """Compute win rate from trade PnL records.

    Args:
        trades: DataFrame with 'pnl' column.

    Returns:
        Fraction of winning trades.
    """
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    wins = (trades["pnl"] > 0).sum()
    return float(wins / len(trades))


def compute_profit_factor(trades: pd.DataFrame) -> float:
    """Compute profit factor (gross profit / gross loss).

    Args:
        trades: DataFrame with 'pnl' column.

    Returns:
        Profit factor, or 0.0 if no losses.
    """
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if gross_loss == 0:
        return float(gross_profit) if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def compute_turnover(target_weights: pd.DataFrame) -> float:
    """Mean one-way turnover across rebalance dates.

    Turnover on a rebalance date is ``0.5 * sum(|Δw|)`` so a full book flip
    from +50%/-50% to the opposite book counts as 1.0.

    Args:
        target_weights: Date x symbol matrix; NaN means hold (ignored). Only
            rows with at least one non-NaN weight are treated as rebalances.

    Returns:
        Average turnover across rebalance dates, or 0.0 if none.
    """
    if target_weights.empty:
        return 0.0

    rebalance = target_weights.dropna(how="all")
    if rebalance.empty:
        return 0.0

    filled = rebalance.fillna(0.0)
    # Assume flat book before the first rebalance.
    previous = pd.Series(0.0, index=filled.columns)
    turnovers: list[float] = []
    for _, row in filled.iterrows():
        delta = (row - previous).abs().sum()
        turnovers.append(float(0.5 * delta))
        previous = row
    return float(np.mean(turnovers)) if turnovers else 0.0


def compute_backtest_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    benchmark_equity: pd.Series | None = None,
    turnover: float = 0.0,
) -> BacktestMetrics:
    """Compute full backtest metrics from equity curve and trades.

    Args:
        equity: Portfolio equity curve.
        trades: Trade records with pnl column.
        benchmark_equity: Optional benchmark equity curve.
        turnover: Average portfolio turnover on rebalance dates.

    Returns:
        BacktestMetrics dataclass.
    """
    returns = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0

    benchmark_return = 0.0
    if benchmark_equity is not None and len(benchmark_equity) > 1:
        benchmark_return = float(benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1)

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=compute_annualized_return(equity),
        sharpe_ratio=compute_sharpe_ratio(returns),
        max_drawdown=compute_max_drawdown(equity),
        win_rate=compute_win_rate(trades),
        profit_factor=compute_profit_factor(trades),
        total_trades=len(trades),
        benchmark_return=benchmark_return,
        turnover=turnover,
    )


def metrics_to_dict(metrics: BacktestMetrics) -> dict:
    """Convert BacktestMetrics to dictionary.

    Args:
        metrics: BacktestMetrics instance.

    Returns:
        Dictionary representation.
    """
    return asdict(metrics)
