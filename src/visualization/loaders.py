"""Data access helpers for the visualization report.

All functions here are read-only: they load artifacts already written by
``python main.py train`` / ``python main.py backtest`` (eval reports, prediction
parquets, the equity curve, and backtest metrics) and normalize them into
tidy pandas structures the plotting functions can consume directly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.data.storage import load_raw_bars
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import LOG_DIR, MODEL_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)

_EVAL_FILENAME_RE = re.compile(r"^eval_(\d{8}_\d{6})_(.+)_(train|val|test)$")


def discover_run_ids() -> list[str]:
    """List distinct run identifiers found among saved eval reports.

    Returns:
        Sorted (chronological) list of run_id strings, e.g. ``20260722_163306``.
    """
    run_ids: set[str] = set()
    for path in LOG_DIR.glob("eval_*.json"):
        match = _EVAL_FILENAME_RE.match(path.stem)
        if match:
            run_ids.add(match.group(1))
    return sorted(run_ids)


def latest_run_id() -> str | None:
    """Return the most recent run_id present in logs/, or None if none exist."""
    run_ids = discover_run_ids()
    return run_ids[-1] if run_ids else None


def load_eval_reports(run_id: str | None = None) -> pd.DataFrame:
    """Load per-symbol, per-split classification eval reports into a tidy frame.

    Args:
        run_id: Specific run to load. Defaults to the most recent run found
            in ``logs/eval_*.json``.

    Returns:
        DataFrame with one row per symbol/split combination and columns
        ``symbol, split, accuracy, precision, recall, f1, roc_auc,
        confusion_matrix, support, run_id``.

    Raises:
        FileNotFoundError: If no eval reports exist (run_id resolves to None)
            or the requested run_id has no matching files.
    """
    resolved_run_id = run_id or latest_run_id()
    if resolved_run_id is None:
        raise FileNotFoundError(
            "No eval reports found under logs/. Run 'python main.py train' first."
        )

    rows: list[dict] = []
    for path in sorted(LOG_DIR.glob(f"eval_{resolved_run_id}_*.json")):
        match = _EVAL_FILENAME_RE.match(path.stem)
        if not match or match.group(1) != resolved_run_id:
            continue
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        rows.append({
            "run_id": resolved_run_id,
            "symbol": report.get("symbol", match.group(2)),
            "split": report.get("data_split", match.group(3)),
            "accuracy": report["accuracy"],
            "precision": report["precision"],
            "recall": report["recall"],
            "f1": report["f1"],
            "roc_auc": report["roc_auc"],
            "pr_auc": report.get("pr_auc", 0.0),
            "brier_score": report.get("brier_score", 0.0),
            "ece": report.get("ece", 0.0),
            "calibration_bins": report.get("calibration_bins", []),
            "reliability_bins": report.get("reliability_bins", []),
            "confusion_matrix": report["confusion_matrix"],
            "support": report["support"],
        })

    if not rows:
        raise FileNotFoundError(f"No eval reports found for run_id={resolved_run_id!r}")

    return pd.DataFrame(rows)


def load_predictions(split: str) -> pd.DataFrame:
    """Load row-level predictions for a split.

    Args:
        split: One of "train", "val", or "test".

    Returns:
        DataFrame with columns ``date, symbol, close, probability, prediction, label``.

    Raises:
        FileNotFoundError: If the split's predictions parquet does not exist.
    """
    path = PROCESSED_DATA_DIR / f"predictions_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found for split={split!r}: {path}. "
            "Run 'python main.py train' first."
        )
    return pd.read_parquet(path)


def load_feature_importance() -> dict[str, float] | None:
    """Load averaged per-feature importance from the latest model manifest.

    Returns:
        Mapping of feature name to averaged importance (sorted descending), or
        None if no model manifest is available.
    """
    path = MODEL_DIR / "latest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    importance = manifest.get("metrics", {}).get("feature_importance")
    return importance or None


def load_equity_curve() -> pd.Series:
    """Load the strategy's daily equity curve saved by the backtest engine.

    Returns:
        Series of equity values indexed by date, named "equity".

    Raises:
        FileNotFoundError: If logs/equity_curve.csv does not exist.
    """
    path = LOG_DIR / "equity_curve.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Equity curve not found: {path}. Run 'python main.py backtest' first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = df["equity"]
    series.index.name = "date"
    series.name = "equity"
    return series


def load_benchmark_equity(config: AppConfig, equity_index: pd.DatetimeIndex) -> pd.Series | None:
    """Recompute the buy-and-hold benchmark equity curve for comparison.

    Mirrors the normalization performed in ``src.backtesting.engine.run_backtest``:
    the benchmark symbol's close price is reindexed onto the strategy's equity
    dates (forward-filled) and scaled to start at the same initial cash.

    Args:
        config: Application configuration (used for benchmark_symbol and initial_cash).
        equity_index: Date index of the strategy equity curve to align to.

    Returns:
        Benchmark equity Series aligned to equity_index, or None if the
        benchmark symbol's raw bars are not cached locally.
    """
    benchmark_symbol = config.backtest.benchmark_symbol
    bars = load_raw_bars(benchmark_symbol)
    if bars is None or bars.empty:
        logger.warning("No cached raw bars for benchmark symbol %s", benchmark_symbol)
        return None

    bench_prices = bars["close"].reindex(equity_index, method="ffill")
    if bench_prices.isna().all():
        return None
    bench_equity = config.backtest.initial_cash * (bench_prices / bench_prices.iloc[0])
    bench_equity.name = "benchmark_equity"
    return bench_equity


def load_backtest_metrics() -> dict:
    """Load portfolio-level backtest metrics.

    Returns:
        Dict with total_return, sharpe_ratio, max_drawdown, win_rate,
        profit_factor, total_trades, benchmark_return.

    Raises:
        FileNotFoundError: If logs/backtest_metrics.json does not exist.
    """
    path = LOG_DIR / "backtest_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Backtest metrics not found: {path}. Run 'python main.py backtest' first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_figures_dir(run_id: str | None) -> Path:
    """Resolve the output directory for a report run.

    Args:
        run_id: Run identifier, or None to use the latest available run.

    Returns:
        Path to logs/figures/{run_id}.
    """
    resolved_run_id = run_id or latest_run_id() or "latest"
    return LOG_DIR / "figures" / resolved_run_id
