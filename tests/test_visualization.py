"""Tests for the train/test and backtest visualization report."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import src.visualization.loaders as loaders
from src.utils.config import AppConfig, BacktestConfig
from src.visualization import backtest_plots, classification_plots, experiment1_plots, experiment2_plots
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


def _sample_experiment1_metrics() -> pd.DataFrame:
    rows = []
    for i, model_type in enumerate(("xgboost", "lightgbm", "random_forest", "catboost")):
        for j, split in enumerate(("train", "val", "test")):
            rows.append({
                "model_type": model_type,
                "split": split,
                "run_id": "20260101_000000",
                "rmse": 0.03 + 0.002 * j + 0.001 * i,
                "mae": 0.02 + 0.001 * j,
                "r2": 0.08 - 0.02 * j - 0.005 * i,
                "roc_auc": 0.58 - 0.01 * j + 0.005 * (2 - i),
                "pr_auc": 0.26 - 0.01 * j,
                "support": 1000,
            })
    return pd.DataFrame(rows)


def _sample_experiment1_cross_sectional() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model_type": "xgboost",
            "split": "test",
            "ic_overall": 0.03,
            "ic_mean_daily": 0.02,
            "ic_std_daily": 0.2,
            "ic_ir": 0.10,
            "top_decile_hit_rate": 0.30,
        },
        {
            "model_type": "lightgbm",
            "split": "test",
            "ic_overall": 0.04,
            "ic_mean_daily": 0.03,
            "ic_std_daily": 0.22,
            "ic_ir": 0.136,
            "top_decile_hit_rate": 0.33,
        },
        {
            "model_type": "random_forest",
            "split": "test",
            "ic_overall": 0.042,
            "ic_mean_daily": 0.035,
            "ic_std_daily": 0.23,
            "ic_ir": 0.152,
            "top_decile_hit_rate": 0.332,
        },
        {
            "model_type": "catboost",
            "split": "test",
            "ic_overall": 0.038,
            "ic_mean_daily": 0.028,
            "ic_std_daily": 0.21,
            "ic_ir": 0.133,
            "top_decile_hit_rate": 0.325,
        },
    ])


def _sample_experiment1_trading() -> pd.DataFrame:
    rows = []
    for i, model_type in enumerate(("xgboost", "lightgbm", "random_forest", "catboost")):
        for split, base in (("val", 0.4), ("test", 0.5)):
            rows.append({
                "model_type": model_type,
                "split": split,
                "run_id": "20260101_000000",
                "annualized_return": 0.05 + 0.01 * i,
                "sharpe_ratio": base + 0.1 * i,
                "max_drawdown": 0.12 - 0.01 * i,
                "win_rate": 0.52 + 0.01 * i,
                "profit_factor": 1.1 + 0.05 * i,
                "turnover": 0.35,
                "total_return": 0.08 + 0.01 * i,
                "total_trades": 80,
            })
    return pd.DataFrame(rows)


def _sample_experiment1_manifest(model_type: str, *, atr: float) -> dict:
    deciles = [
        {"decile": d, "mean_forward_return": -0.01 + 0.002 * d, "count": 100}
        for d in range(1, 11)
    ]
    split_block = {
        "rmse": 0.035,
        "mae": 0.025,
        "r2": 0.02,
        "roc_auc": 0.58,
        "pr_auc": 0.26,
        "support": 1000,
        "cross_sectional": {
            "ic_overall": 0.04,
            "ic_mean_daily": 0.03,
            "ic_std_daily": 0.2,
            "top_decile_hit_rate": 0.33,
            "mean_return_by_prediction_decile": deciles,
        },
    }
    return {
        "scope": "pooled",
        "model_type": model_type,
        "run_id": "20260101_000000",
        "artifacts": [
            {
                "symbol": "pooled",
                "model_type": model_type,
                "train_rows": 1000,
                "val_rows": 500,
                "test_rows": 400,
                "feature_columns": ["atr_pct", "volatility_20d", "bb_width"],
                "metrics": {
                    "task": "regression",
                    "target": "forward_return_5d",
                    "train": split_block,
                    "val": split_block,
                    "test": split_block,
                    "feature_importance": {
                        "atr_pct": atr,
                        "volatility_20d": 0.15,
                        "bb_width": 0.08,
                        "rsi_14": 0.05,
                    },
                },
            }
        ],
    }


def _write_experiment1_fixture(results_dir) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    _sample_experiment1_metrics().to_csv(results_dir / "metrics_by_model_split.csv", index=False)
    _sample_experiment1_cross_sectional().to_csv(
        results_dir / "cross_sectional_by_model.csv", index=False
    )
    _sample_experiment1_trading().to_csv(results_dir / "trading_by_model.csv", index=False)
    with (results_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "winner": "random_forest",
            "task": "regression",
            "winner_rule": "test_sharpe -> ic_mean_daily -> top_decile_hit_rate",
            "runs": [],
        }, handle)
    for model_type, atr in (
        ("xgboost", 0.17),
        ("lightgbm", 0.20),
        ("random_forest", 0.27),
        ("catboost", 0.22),
    ):
        model_dir = results_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "model_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(_sample_experiment1_manifest(model_type, atr=atr), handle)


class TestExperiment1Plots:
    """Smoke tests for Experiment 1 multi-model comparison figures."""

    def test_plot_model_comparison_test_metrics(self, tmp_path) -> None:
        out_path = tmp_path / "test_metrics.png"
        result = experiment1_plots.plot_model_comparison_test_metrics(
            _sample_experiment1_metrics(),
            _sample_experiment1_cross_sectional(),
            out_path,
        )
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_model_comparison_metrics_by_split(self, tmp_path) -> None:
        out_path = tmp_path / "by_split.png"
        experiment1_plots.plot_model_comparison_metrics_by_split(
            _sample_experiment1_metrics(), out_path
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_model_comparison_cross_sectional(self, tmp_path) -> None:
        out_path = tmp_path / "cross_sectional.png"
        experiment1_plots.plot_model_comparison_cross_sectional(
            _sample_experiment1_cross_sectional(), out_path
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_model_comparison_feature_importance(self, tmp_path) -> None:
        out_path = tmp_path / "feature_importance.png"
        importances = {
            "xgboost": {"atr_pct": 0.17, "volatility_20d": 0.09, "bb_width": 0.05},
            "lightgbm": {"atr_pct": 0.20, "volatility_20d": 0.10, "bb_width": 0.06},
            "random_forest": {"atr_pct": 0.27, "volatility_20d": 0.14, "bb_width": 0.07},
            "catboost": {"atr_pct": 0.22, "volatility_20d": 0.12, "bb_width": 0.06},
        }
        experiment1_plots.plot_model_comparison_feature_importance(importances, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_model_comparison_prediction_deciles(self, tmp_path) -> None:
        out_path = tmp_path / "deciles.png"
        rows = []
        for model_type in ("xgboost", "lightgbm", "random_forest", "catboost"):
            for d in range(1, 11):
                rows.append({
                    "model_type": model_type,
                    "split": "test",
                    "decile": d,
                    "mean_forward_return": -0.01 + 0.002 * d,
                    "count": 100,
                })
        experiment1_plots.plot_model_comparison_prediction_deciles(pd.DataFrame(rows), out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_model_comparison_returns(self, tmp_path) -> None:
        out_path = tmp_path / "returns.png"
        rows = []
        for model_type in ("xgboost", "lightgbm", "random_forest", "catboost"):
            for d in range(1, 11):
                rows.append({
                    "model_type": model_type,
                    "split": "test",
                    "decile": d,
                    "mean_forward_return": -0.01 + 0.002 * d,
                    "count": 100,
                })
        experiment1_plots.plot_model_comparison_returns(pd.DataFrame(rows), out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_model_comparison_trading(self, tmp_path) -> None:
        out_path = tmp_path / "trading.png"
        experiment1_plots.plot_model_comparison_trading(
            _sample_experiment1_trading(), out_path, split="test"
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_generate_experiment1_figures_end_to_end(self, tmp_path) -> None:
        results_dir = tmp_path / "experiment_1"
        _write_experiment1_fixture(results_dir)

        written = experiment1_plots.generate_experiment1_figures(
            results_dir, regenerate_per_model=True
        )

        figures_dir = results_dir / "figures"
        expected = {
            "model_comparison_test_metrics.png",
            "model_comparison_metrics_by_split.png",
            "model_comparison_cross_sectional.png",
            "model_comparison_feature_importance.png",
            "model_comparison_prediction_deciles.png",
            "model_comparison_returns.png",
            "model_comparison_trading.png",
            "feature_importance_winner.png",
        }
        written_names = {path.name for path in written if path.parent == figures_dir}
        assert expected.issubset(written_names)
        assert all((figures_dir / name).stat().st_size > 0 for name in expected)
        # Regression manifests skip classification 01/02; feature importance still written.
        for model_type in ("xgboost", "lightgbm", "random_forest", "catboost"):
            model_figs = results_dir / model_type / "figures"
            assert (model_figs / "05_feature_importance.png").exists()
            assert not (model_figs / "01_classification_metric_overview.png").exists()


def _sample_experiment2_metrics() -> pd.DataFrame:
    rows = []
    arms = [
        ("full", "A", 19),
        ("returns", "A", 3),
        ("volatility", "A", 3),
        ("top5", "B", 5),
    ]
    for i, (arm, stage, n_features) in enumerate(arms):
        for j, split in enumerate(("train", "val", "test")):
            rows.append({
                "arm": arm,
                "stage": stage,
                "split": split,
                "run_id": "20260101_000000",
                "n_features": n_features,
                "rmse": 0.03 + 0.001 * i,
                "mae": 0.02,
                "r2": 0.05 - 0.01 * j,
                "roc_auc": 0.52 + 0.01 * i,
                "pr_auc": 0.24,
                "support": 1000,
            })
    return pd.DataFrame(rows)


def _sample_experiment2_cross_sectional() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "arm": "full",
            "stage": "A",
            "split": "test",
            "ic_overall": 0.02,
            "ic_mean_daily": 0.0,
            "ic_std_daily": 0.18,
            "ic_ir": 0.0,
            "top_decile_hit_rate": 0.29,
        },
        {
            "arm": "returns",
            "stage": "A",
            "split": "test",
            "ic_overall": 0.025,
            "ic_mean_daily": 0.004,
            "ic_std_daily": 0.19,
            "ic_ir": 0.021,
            "top_decile_hit_rate": 0.30,
        },
        {
            "arm": "volatility",
            "stage": "A",
            "split": "test",
            "ic_overall": 0.03,
            "ic_mean_daily": 0.008,
            "ic_std_daily": 0.2,
            "ic_ir": 0.04,
            "top_decile_hit_rate": 0.31,
        },
        {
            "arm": "top5",
            "stage": "B",
            "split": "test",
            "ic_overall": 0.028,
            "ic_mean_daily": 0.006,
            "ic_std_daily": 0.19,
            "ic_ir": 0.032,
            "top_decile_hit_rate": 0.305,
        },
    ])


def _sample_experiment2_trading() -> pd.DataFrame:
    rows = []
    for i, (arm, stage) in enumerate(
        (("full", "A"), ("returns", "A"), ("volatility", "A"), ("top5", "B"))
    ):
        for split, base in (("val", 0.5), ("test", -0.1)):
            rows.append({
                "arm": arm,
                "stage": stage,
                "split": split,
                "run_id": "20260101_000000",
                "n_features": 5,
                "annualized_return": -0.02 + 0.01 * i,
                "sharpe_ratio": base + 0.05 * i,
                "max_drawdown": 0.1,
                "win_rate": 0.48,
                "profit_factor": 0.95,
                "turnover": 0.65,
                "total_return": -0.03,
                "total_trades": 1000,
            })
    return pd.DataFrame(rows)


def _write_experiment2_fixture(results_dir) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    _sample_experiment2_metrics().to_csv(results_dir / "metrics_by_arm_split.csv", index=False)
    _sample_experiment2_cross_sectional().to_csv(
        results_dir / "cross_sectional_by_arm.csv", index=False
    )
    _sample_experiment2_trading().to_csv(results_dir / "trading_by_arm.csv", index=False)
    with (results_dir / "full_feature_importance.json").open("w", encoding="utf-8") as handle:
        json.dump({"atr_pct": 0.25, "volatility_20d": 0.2, "return_5d": 0.15}, handle)
    with (results_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "winner": "volatility",
            "frozen_model": "xgboost",
            "best_stage_b_by_val_sharpe": "top5",
            "runs": [],
        }, handle)


class TestExperiment2Plots:
    """Smoke tests for Experiment 2 feature-arm comparison figures."""

    def test_plot_arm_comparison_test_metrics(self, tmp_path) -> None:
        out_path = tmp_path / "test_metrics.png"
        result = experiment2_plots.plot_arm_comparison_test_metrics(
            _sample_experiment2_metrics(),
            _sample_experiment2_cross_sectional(),
            out_path,
        )
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_arm_comparison_trading(self, tmp_path) -> None:
        out_path = tmp_path / "trading.png"
        experiment2_plots.plot_arm_comparison_trading(
            _sample_experiment2_trading(), out_path, split="test"
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_generate_experiment2_figures_end_to_end(self, tmp_path) -> None:
        results_dir = tmp_path / "experiment_2"
        _write_experiment2_fixture(results_dir)

        written = experiment2_plots.generate_experiment2_figures(results_dir)

        figures_dir = results_dir / "figures"
        expected = {
            "arm_comparison_test_metrics.png",
            "arm_comparison_cross_sectional.png",
            "arm_comparison_trading.png",
            "full_feature_importance.png",
        }
        written_names = {path.name for path in written}
        assert expected.issubset(written_names)
        assert all((figures_dir / name).stat().st_size > 0 for name in expected)
