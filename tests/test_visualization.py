"""Tests for the train/test and backtest visualization report."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import src.visualization.loaders as loaders
from src.utils.config import AppConfig, BacktestConfig
from src.visualization import backtest_plots, classification_plots
from src.visualization.report import generate_report


def _write_eval_json(log_dir, run_id: str, symbol: str, split: str, **overrides) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "accuracy": 0.55,
        "precision": 0.6,
        "recall": 0.4,
        "f1": 0.48,
        "roc_auc": 0.58,
        "confusion_matrix": [[10, 5], [4, 11]],
        "support": 30,
        "run_id": run_id,
        "symbol": symbol,
        "data_split": split,
    }
    report.update(overrides)
    path = log_dir / f"eval_{run_id}_{symbol}_{split}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle)


def _sample_eval_df() -> pd.DataFrame:
    rows = []
    for symbol in ("AAA", "BBB", "CCC"):
        for split in ("val", "test"):
            rows.append({
                "run_id": "20260101_000000",
                "symbol": symbol,
                "split": split,
                "accuracy": 0.55,
                "precision": 0.6,
                "recall": 0.4,
                "f1": 0.48,
                "roc_auc": 0.58 if split == "val" else 0.51,
                "confusion_matrix": [[10, 5], [4, 11]],
                "support": 30,
            })
    return pd.DataFrame(rows)


def _sample_predictions_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6, freq="D"),
        "symbol": ["AAA"] * 6,
        "close": [100.0] * 6,
        "probability": [0.2, 0.8, 0.6, 0.3, 0.9, 0.1],
        "probability_platt": [0.15, 0.7, 0.5, 0.25, 0.8, 0.08],
        "probability_isotonic": [0.18, 0.75, 0.55, 0.28, 0.85, 0.09],
        "prediction": [0, 1, 1, 0, 1, 0],
        "label": [0, 1, 1, 0, 1, 0],
    })


def _sample_equity() -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=40, freq="B")
    series = pd.Series([100_000.0 * (1 + 0.001 * i) for i in range(40)], index=idx, name="equity")
    series.index.name = "date"
    return series


class TestLoaders:
    """Tests for src.visualization.loaders."""

    def test_discover_and_latest_run_id(self, tmp_path, monkeypatch) -> None:
        """Run ids are parsed from eval filenames and sorted chronologically."""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(loaders, "LOG_DIR", log_dir)
        _write_eval_json(log_dir, "20260101_000000", "AAA", "val")
        _write_eval_json(log_dir, "20260202_000000", "AAA", "val")

        assert loaders.discover_run_ids() == ["20260101_000000", "20260202_000000"]
        assert loaders.latest_run_id() == "20260202_000000"

    def test_latest_run_id_none_when_no_reports(self, tmp_path, monkeypatch) -> None:
        """No eval reports means no run id is discoverable."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        monkeypatch.setattr(loaders, "LOG_DIR", log_dir)

        assert loaders.discover_run_ids() == []
        assert loaders.latest_run_id() is None

    def test_load_eval_reports_defaults_to_latest_run(self, tmp_path, monkeypatch) -> None:
        """With no run_id given, the most recent run's reports are loaded."""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(loaders, "LOG_DIR", log_dir)
        _write_eval_json(log_dir, "20260101_000000", "AAA", "val", accuracy=0.1)
        _write_eval_json(log_dir, "20260202_000000", "AAA", "val", accuracy=0.9)
        _write_eval_json(log_dir, "20260202_000000", "AAA", "test", accuracy=0.8)

        df = loaders.load_eval_reports()

        assert set(df["run_id"]) == {"20260202_000000"}
        assert set(df["split"]) == {"val", "test"}
        assert df.loc[df["split"] == "val", "accuracy"].iloc[0] == pytest.approx(0.9)

    def test_load_eval_reports_explicit_run_id(self, tmp_path, monkeypatch) -> None:
        """An explicit run_id selects that run's reports, not the latest one."""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(loaders, "LOG_DIR", log_dir)
        _write_eval_json(log_dir, "20260101_000000", "AAA", "val", accuracy=0.1)
        _write_eval_json(log_dir, "20260202_000000", "AAA", "val", accuracy=0.9)

        df = loaders.load_eval_reports(run_id="20260101_000000")

        assert set(df["run_id"]) == {"20260101_000000"}
        assert df["accuracy"].iloc[0] == pytest.approx(0.1)

    def test_load_eval_reports_handles_symbols_with_dots(self, tmp_path, monkeypatch) -> None:
        """Tickers like BRK.B are parsed correctly out of the filename."""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(loaders, "LOG_DIR", log_dir)
        _write_eval_json(log_dir, "20260101_000000", "BRK.B", "test")

        df = loaders.load_eval_reports()

        assert df["symbol"].iloc[0] == "BRK.B"
        assert df["split"].iloc[0] == "test"

    def test_load_eval_reports_missing_raises(self, tmp_path, monkeypatch) -> None:
        """No eval reports on disk raises a clear FileNotFoundError."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        monkeypatch.setattr(loaders, "LOG_DIR", log_dir)

        with pytest.raises(FileNotFoundError):
            loaders.load_eval_reports()

    def test_load_feature_importance_missing_returns_none(self, tmp_path, monkeypatch) -> None:
        """No model manifest means feature importance is unavailable, not an error."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        monkeypatch.setattr(loaders, "MODEL_DIR", model_dir)

        assert loaders.load_feature_importance() is None

    def test_load_benchmark_equity_normalizes_to_initial_cash(self, tmp_path, monkeypatch) -> None:
        """Benchmark equity is reindexed and scaled to the strategy's starting cash."""
        equity_index = pd.date_range("2024-01-01", periods=5, freq="D")
        bars = pd.DataFrame(
            {"close": [10.0, 11.0, 12.0, 13.0, 14.0]},
            index=equity_index,
        )
        monkeypatch.setattr(loaders, "load_raw_bars", lambda symbol: bars)

        config = AppConfig(backtest=BacktestConfig(initial_cash=1_000.0, benchmark_symbol="SPY"))
        benchmark = loaders.load_benchmark_equity(config, equity_index)

        assert benchmark is not None
        assert benchmark.iloc[0] == pytest.approx(1_000.0)
        assert benchmark.iloc[-1] == pytest.approx(1_000.0 * 14.0 / 10.0)

    def test_load_benchmark_equity_missing_bars_returns_none(self, tmp_path, monkeypatch) -> None:
        """Missing benchmark cache degrades gracefully instead of raising."""
        monkeypatch.setattr(loaders, "load_raw_bars", lambda symbol: None)
        config = AppConfig(backtest=BacktestConfig())

        result = loaders.load_benchmark_equity(config, pd.date_range("2024-01-01", periods=3))

        assert result is None


class TestClassificationPlots:
    """Smoke tests that classification figures render without error."""

    def test_plot_metric_overview(self, tmp_path) -> None:
        out_path = tmp_path / "metric_overview.png"
        result = classification_plots.plot_metric_overview(_sample_eval_df(), out_path)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_roc_auc_distribution(self, tmp_path) -> None:
        out_path = tmp_path / "roc_auc.png"
        classification_plots.plot_roc_auc_distribution(_sample_eval_df(), out_path)
        assert out_path.exists()

    def test_plot_confusion_matrices(self, tmp_path) -> None:
        out_path = tmp_path / "confusion.png"
        classification_plots.plot_confusion_matrices(_sample_eval_df(), out_path)
        assert out_path.exists()

    def test_plot_probability_separation(self, tmp_path) -> None:
        out_path = tmp_path / "prob_sep.png"
        classification_plots.plot_probability_separation(
            _sample_predictions_df(), _sample_predictions_df(), out_path
        )
        assert out_path.exists()

    def test_plot_feature_importance(self, tmp_path) -> None:
        out_path = tmp_path / "feature_importance.png"
        importances = {f"feature_{i}": float(20 - i) for i in range(20)}
        classification_plots.plot_feature_importance(importances, out_path, top_n=15)
        assert out_path.exists()

    def test_plot_calibration_curves(self, tmp_path) -> None:
        out_path = tmp_path / "calibration.png"
        pred = _sample_predictions_df()
        classification_plots.plot_calibration_curves(
            {"val": pred, "test": pred}, out_path
        )
        assert out_path.exists()


class TestBacktestPlots:
    """Smoke tests that backtest figures render without error."""

    def test_plot_equity_and_drawdown_with_benchmark(self, tmp_path) -> None:
        equity = _sample_equity()
        benchmark = equity * 0.98
        out_path = tmp_path / "equity.png"
        backtest_plots.plot_equity_and_drawdown(equity, benchmark, out_path)
        assert out_path.exists()

    def test_plot_equity_and_drawdown_without_benchmark(self, tmp_path) -> None:
        out_path = tmp_path / "equity_no_bench.png"
        backtest_plots.plot_equity_and_drawdown(_sample_equity(), None, out_path)
        assert out_path.exists()

    def test_plot_rolling_sharpe(self, tmp_path) -> None:
        out_path = tmp_path / "rolling_sharpe.png"
        backtest_plots.plot_rolling_sharpe(_sample_equity(), out_path, window=5)
        assert out_path.exists()

    def test_plot_backtest_scorecard(self, tmp_path) -> None:
        out_path = tmp_path / "scorecard.png"
        metrics = {
            "total_return": 0.12,
            "sharpe_ratio": 1.1,
            "max_drawdown": 0.08,
            "win_rate": 0.55,
            "profit_factor": 1.3,
            "total_trades": 120,
            "benchmark_return": 0.09,
        }
        backtest_plots.plot_backtest_scorecard(metrics, out_path)
        assert out_path.exists()


class TestGenerateReport:
    """End-to-end smoke tests for the report orchestrator."""

    def test_generate_report_skips_gracefully_when_nothing_available(
        self, tmp_path, monkeypatch
    ) -> None:
        """With no artifacts on disk, the report should skip every figure, not raise."""
        log_dir = tmp_path / "logs"
        model_dir = tmp_path / "models"
        processed_dir = tmp_path / "processed"
        log_dir.mkdir()
        model_dir.mkdir()
        processed_dir.mkdir()
        monkeypatch.setattr(loaders, "LOG_DIR", log_dir)
        monkeypatch.setattr(loaders, "MODEL_DIR", model_dir)
        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", processed_dir)

        result = generate_report(AppConfig(), run_id=None)

        assert result.written == []
        assert len(result.skipped) > 0

    def test_generate_report_writes_available_figures(self, tmp_path, monkeypatch) -> None:
        """Available artifacts produce their figures while missing ones are skipped."""
        log_dir = tmp_path / "logs"
        model_dir = tmp_path / "models"
        processed_dir = tmp_path / "processed"
        log_dir.mkdir()
        model_dir.mkdir()
        processed_dir.mkdir()
        monkeypatch.setattr(loaders, "LOG_DIR", log_dir)
        monkeypatch.setattr(loaders, "MODEL_DIR", model_dir)
        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(loaders, "load_raw_bars", lambda symbol: None)

        for symbol in ("AAA", "BBB"):
            for split in ("val", "test"):
                _write_eval_json(log_dir, "20260101_000000", symbol, split)

        _sample_predictions_df().to_parquet(processed_dir / "predictions_val.parquet", index=False)
        _sample_predictions_df().to_parquet(processed_dir / "predictions_test.parquet", index=False)

        equity_df = _sample_equity().to_frame()
        equity_df.to_csv(log_dir / "equity_curve.csv")

        with (log_dir / "backtest_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.05,
                "win_rate": 0.5,
                "profit_factor": 1.2,
                "total_trades": 50,
                "benchmark_return": 0.08,
            }, handle)

        result = generate_report(AppConfig(), run_id=None)

        assert len(result.written) == 8
        assert all(path.exists() for path in result.written)
        assert result.skipped == [
            "feature importance (05): models/latest.json not found or has no feature_importance"
        ]
        assert result.output_dir == log_dir / "figures" / "20260101_000000"
