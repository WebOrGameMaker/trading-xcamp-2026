"""Vectorbt backtesting engine for the weekly long/short rebalance strategy."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.backtesting.metrics import (
    BacktestMetrics,
    compute_backtest_metrics,
    compute_turnover,
    metrics_to_dict,
)
from src.data.storage import load_raw_bars
from src.strategy.portfolio import build_portfolio_signals
from src.strategy.rebalance import weekly_rebalance_dates
from src.utils.config import AppConfig, StrategyConfig
from src.utils.logging import get_logger
from src.utils.paths import LOG_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)


def _build_price_matrix(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Pivot close prices into a date x symbol matrix.

    Args:
        portfolio: Portfolio signals with date, symbol, close.

    Returns:
        Wide DataFrame of close prices.
    """
    prices = portfolio.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    return prices.sort_index()


def _build_target_weight_matrix(
    portfolio: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a date x symbol target-percent matrix for weekly rebalancing.

    Non-rebalance days are left as NaN (hold current position). On each
    rebalance date every symbol gets an explicit target weight (0.0 if not
    selected that week), so positions dropped from the top/bottom ranks are
    actively closed out rather than held indefinitely.

    Args:
        portfolio: Portfolio signals with date, symbol, weight columns.
        rebalance_dates: Dates on which the book is rebalanced.

    Returns:
        Wide DataFrame of target weights (NaN = hold, else target percent).
    """
    all_dates = sorted(portfolio["date"].unique())
    all_symbols = sorted(portfolio["symbol"].unique())

    weights = portfolio.pivot_table(index="date", columns="symbol", values="weight", aggfunc="last")
    weights = weights.reindex(index=all_dates, columns=all_symbols)

    target = pd.DataFrame(
        np.nan,
        index=pd.Index(all_dates, name="date"),
        columns=pd.Index(all_symbols, name="symbol"),
    )
    valid_rebalance_dates = pd.DatetimeIndex(
        [d for d in rebalance_dates if d in target.index]
    )
    if len(valid_rebalance_dates) > 0:
        target.loc[valid_rebalance_dates] = weights.loc[valid_rebalance_dates].fillna(0.0)

    return target


def run_backtest_on_predictions(
    config: AppConfig,
    predictions: pd.DataFrame,
    strategy: StrategyConfig | None = None,
    *,
    persist: bool = False,
    equity_path: Path | None = None,
    metrics_path: Path | None = None,
) -> BacktestMetrics:
    """Run a weekly long/short vectorbt backtest on an in-memory predictions frame.

    Args:
        config: Application configuration (backtest cash/fees/benchmark).
        predictions: Prediction rows with date, symbol, close, and probability columns.
        strategy: Strategy settings override; defaults to ``config.strategy``.
        persist: When True, write equity curve and metrics JSON (default CLI path).
        equity_path: Optional equity CSV destination when persisting.
        metrics_path: Optional metrics JSON destination when persisting.

    Returns:
        BacktestMetrics with performance statistics.
    """
    strategy = strategy or config.strategy
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    portfolio = build_portfolio_signals(frame, strategy)
    prices = _build_price_matrix(portfolio)

    rebalance_dates = weekly_rebalance_dates(portfolio["date"].unique())
    target_weights = _build_target_weight_matrix(portfolio, rebalance_dates)

    common_cols = prices.columns.intersection(target_weights.columns)
    prices = prices[common_cols]
    target_weights = target_weights[common_cols]

    fees = (config.backtest.commission_bps + config.backtest.slippage_bps) / 10_000

    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=target_weights,
        size_type="targetpercent",
        direction="both",
        group_by=True,
        cash_sharing=True,
        init_cash=config.backtest.initial_cash,
        fees=fees,
        freq="1D",
    )

    equity = pf.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.sum(axis=1)

    trades_records = pf.trades.records_readable.rename(columns={"PnL": "pnl"})
    turnover = compute_turnover(target_weights)
    metrics = compute_backtest_metrics(equity, trades_records, turnover=turnover)

    benchmark_symbol = config.backtest.benchmark_symbol
    benchmark_bars = load_raw_bars(benchmark_symbol)
    if benchmark_bars is not None:
        bench_prices = benchmark_bars["close"].reindex(equity.index, method="ffill")
        if not bench_prices.isna().all():
            bench_equity = config.backtest.initial_cash * (bench_prices / bench_prices.iloc[0])
            bench_metrics = compute_backtest_metrics(bench_equity, pd.DataFrame())
            metrics = BacktestMetrics(
                total_return=metrics.total_return,
                annualized_return=metrics.annualized_return,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                profit_factor=metrics.profit_factor,
                total_trades=metrics.total_trades,
                benchmark_return=bench_metrics.total_return,
                turnover=metrics.turnover,
            )

    if persist:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        equity_out = equity_path or (LOG_DIR / "equity_curve.csv")
        metrics_out = metrics_path or (LOG_DIR / "backtest_metrics.json")
        equity.to_csv(equity_out, header=["equity"])
        with metrics_out.open("w", encoding="utf-8") as handle:
            json.dump(metrics_to_dict(metrics), handle, indent=2)

    logger.info(
        "Backtest (mode=%s, %d rebalances) — return: %.2f%%, ann: %.2f%%, "
        "sharpe: %.2f, max DD: %.2f%%, win rate: %.2f%%, turnover: %.2f",
        strategy.mode,
        len(rebalance_dates),
        metrics.total_return * 100,
        metrics.annualized_return * 100,
        metrics.sharpe_ratio,
        metrics.max_drawdown * 100,
        metrics.win_rate * 100,
        metrics.turnover,
    )
    return metrics


def run_backtest(
    config: AppConfig,
    *,
    split: str = "test",
    strategy: StrategyConfig | None = None,
    persist: bool = True,
) -> BacktestMetrics:
    """Run a weekly long/short vectorbt backtest on a predictions split.

    Default behavior (``split='test'``, ``persist=True``) matches the historical
    CLI: consume ``predictions_test.parquet`` and write
    ``logs/equity_curve.csv`` + ``logs/backtest_metrics.json``.

    Args:
        config: Application configuration.
        split: Prediction split to load (``train`` / ``val`` / ``test``).
        strategy: Optional strategy override.
        persist: Whether to write default equity/metrics artifacts.

    Returns:
        BacktestMetrics with performance statistics.
    """
    pred_path = PROCESSED_DATA_DIR / f"predictions_{split}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions not found for split={split!r}: {pred_path}. "
            "Run 'python main.py train' first."
        )

    predictions = pd.read_parquet(pred_path)
    return run_backtest_on_predictions(
        config,
        predictions,
        strategy=strategy,
        persist=persist,
    )


def with_strategy_overrides(config: AppConfig, **overrides: object) -> AppConfig:
    """Return a deep-copied config with selected strategy fields replaced."""
    cloned = deepcopy(config)
    cloned.strategy = replace(cloned.strategy, **overrides)
    return cloned
