"""Figures summarizing portfolio-level backtest performance vs the benchmark.

These translate the strategy's simulated trading behavior on the held-out
test period into the handful of charts most reviewers expect: growth vs
buy-and-hold, drawdown risk over time, return consistency, and a compact
scorecard of the headline numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    FIGSIZE_TALL,
    FIGSIZE_WIDE,
    REFERENCE_LINE_COLOR,
    SERIES_COLORS,
    apply_style,
    save_figure,
)


def plot_equity_and_drawdown(
    equity: pd.Series,
    benchmark: pd.Series | None,
    out_path: str | Path,
) -> Path:
    """Two-panel figure: cumulative return vs benchmark, and underwater drawdown.

    Plotting cumulative % return (rather than raw $ equity) keeps the chart
    comparable across strategy runs even if initial cash differs.

    Args:
        equity: Strategy equity curve indexed by date.
        benchmark: Optional benchmark equity curve aligned to the same dates.
        out_path: Destination PNG path.

    Returns:
        The path the figure was saved to.
    """
    apply_style()
    strategy_return = (equity / equity.iloc[0] - 1.0) * 100.0
    strategy_drawdown = _drawdown_series(equity) * 100.0

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=FIGSIZE_TALL, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_top.plot(
        strategy_return.index,
        strategy_return.values,
        color=SERIES_COLORS["strategy"],
        label="Strategy",
        linewidth=1.8,
    )
    if benchmark is not None and not benchmark.dropna().empty:
        benchmark_return = (benchmark / benchmark.iloc[0] - 1.0) * 100.0
        ax_top.plot(
            benchmark_return.index,
            benchmark_return.values,
            color=SERIES_COLORS["benchmark"],
            label="Benchmark (buy & hold)",
            linewidth=1.5,
            linestyle="--",
        )
    ax_top.axhline(0, color=REFERENCE_LINE_COLOR, linewidth=0.8)
    ax_top.set_ylabel("Cumulative return (%)")
    ax_top.set_title("Strategy vs Benchmark — Cumulative Return")
    ax_top.legend(loc="upper left", framealpha=0.9)

    ax_bottom.fill_between(
        strategy_drawdown.index, strategy_drawdown.values, 0,
        color=SERIES_COLORS["negative"], alpha=0.4,
    )
    ax_bottom.plot(
        strategy_drawdown.index, strategy_drawdown.values,
        color=SERIES_COLORS["negative"], linewidth=1,
    )
    ax_bottom.set_ylabel("Drawdown (%)")
    ax_bottom.set_title("Strategy Underwater Drawdown")
    ax_bottom.set_xlabel("Date")
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max.replace(0, np.nan)


def plot_rolling_sharpe(
    equity: pd.Series,
    out_path: str | Path,
    window: int = 60,
    periods_per_year: int = 252,
) -> Path:
    """Rolling annualized Sharpe ratio over time.

    A single full-period Sharpe number can hide whether performance was
    consistent or driven by a short lucky/unlucky stretch; this shows the
    trend over time.

    Args:
        equity: Strategy equity curve indexed by date.
        out_path: Destination PNG path.
        window: Rolling window length in trading days.
        periods_per_year: Annualization factor.

    Returns:
        The path the figure was saved to.
    """
    apply_style()
    returns = equity.pct_change().dropna()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(periods_per_year)).dropna()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(
        rolling_sharpe.index, rolling_sharpe.values,
        color=SERIES_COLORS["strategy"], linewidth=1.5,
    )
    ax.fill_between(
        rolling_sharpe.index, rolling_sharpe.values, 0,
        where=rolling_sharpe.values >= 0, color=SERIES_COLORS["positive"], alpha=0.25,
    )
    ax.fill_between(
        rolling_sharpe.index, rolling_sharpe.values, 0,
        where=rolling_sharpe.values < 0, color=SERIES_COLORS["negative"], alpha=0.25,
    )
    ax.axhline(0, color=REFERENCE_LINE_COLOR, linewidth=0.8)
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_xlabel("Date")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    fig.autofmt_xdate(rotation=30)

    save_figure(fig, out_path)
    return Path(out_path)


def plot_backtest_scorecard(metrics: dict, out_path: str | Path) -> Path:
    """Compact scorecard: strategy vs benchmark return, plus a metrics table.

    Args:
        metrics: Dict as saved in logs/backtest_metrics.json (total_return,
            sharpe_ratio, max_drawdown, win_rate, profit_factor, total_trades,
            benchmark_return).
        out_path: Destination PNG path.

    Returns:
        The path the figure was saved to.
    """
    apply_style()
    fig, (ax_bar, ax_table) = plt.subplots(
        1, 2, figsize=FIGSIZE_WIDE, gridspec_kw={"width_ratios": [1, 1.4]}
    )

    labels = ["Strategy", "Benchmark"]
    returns_pct = [metrics["total_return"] * 100, metrics["benchmark_return"] * 100]
    colors = [SERIES_COLORS["strategy"], SERIES_COLORS["benchmark"]]
    bars = ax_bar.bar(labels, returns_pct, color=colors)
    ax_bar.bar_label(bars, fmt="%.1f%%", padding=3)
    ax_bar.axhline(0, color=REFERENCE_LINE_COLOR, linewidth=0.8)
    ax_bar.set_ylabel("Total return (%)")
    ax_bar.set_title("Total Return")

    ax_table.axis("off")
    rows = [
        ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
        ("Ann. Return", f"{metrics.get('annualized_return', 0) * 100:.2f}%"),
        ("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%"),
        ("Win Rate", f"{metrics['win_rate'] * 100:.2f}%"),
        ("Profit Factor", f"{metrics['profit_factor']:.2f}"),
        ("Turnover", f"{metrics.get('turnover', 0):.2f}"),
        ("Total Trades", f"{metrics['total_trades']:,}"),
    ]
    table = ax_table.table(
        cellText=[[name, value] for name, value in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f0f0f0")
    ax_table.set_title("Trading Performance Metrics", pad=20)

    fig.suptitle("Backtest Scorecard", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path)
    return Path(out_path)
