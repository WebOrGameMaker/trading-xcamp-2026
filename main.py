"""CLI entry point for the AI trading bot pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import ensure_directories

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="AI-powered stock trading system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download",
        help="Download historical market data from yfinance",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when cached parquet exists",
    )
    subparsers.add_parser("features", help="Engineer features and generate labels")
    train_parser = subparsers.add_parser("train", help="Train ML model")
    train_parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "random_forest", "catboost"],
        default=None,
        help="Model type override",
    )
    subparsers.add_parser("backtest", help="Run vectorbt backtest on test period")
    subparsers.add_parser(
        "calibrate",
        help=(
            "Compare raw/Platt/isotonic probabilities and ranking vs "
            "confidence-filtered portfolios (thresholds selected on val only)"
        ),
    )
    paper_parser = subparsers.add_parser("paper-trade", help="Execute paper trades via Alpaca")
    paper_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log orders without submitting to Alpaca",
    )
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    subparsers.add_parser("pipeline", help="Run full pipeline: download, features, train, backtest")
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate presentation-ready train/test and backtest result figures",
    )
    visualize_parser.add_argument(
        "--run-id",
        default=None,
        help="Training run to report on (defaults to the most recent run in logs/)",
    )

    return parser


def cmd_download(config_path: str, force: bool = False) -> None:
    """Download historical OHLCV data."""
    from src.data.downloader import download_universe

    config = load_config(config_path)
    download_universe(config, force=force)


def cmd_features(config_path: str) -> None:
    """Engineer features and labels."""
    from src.features.pipeline import build_feature_dataset

    config = load_config(config_path)
    build_feature_dataset(config)


def cmd_train(config_path: str, model_type: str | None) -> None:
    """Train and evaluate ML model."""
    from src.models.trainer import train_model

    config = load_config(config_path)
    if model_type:
        config.model.type = model_type
    train_model(config)


def cmd_backtest(config_path: str) -> None:
    """Run backtest on held-out test period."""
    from src.backtesting.engine import run_backtest

    config = load_config(config_path)
    run_backtest(config)


def cmd_calibrate(config_path: str) -> None:
    """Evaluate calibration quality and confidence-filtered trading."""
    from src.models.calibration_analysis import run_calibration_analysis

    config = load_config(config_path)
    report = run_calibration_analysis(config)
    rec = report["recommendation"]
    logger.info(
        "Calibration recommendation: mode=%s, probability_column=%s — %s",
        rec["production_mode"],
        rec["production_probability_column"],
        rec["rationale"],
    )


def cmd_paper_trade(config_path: str, dry_run: bool) -> None:
    """Execute paper trades."""
    from src.execution.scheduler import run_paper_trading

    config = load_config(config_path)
    if dry_run:
        config.execution.dry_run = True
    run_paper_trading(config)


def cmd_dashboard(config_path: str) -> None:
    """Launch Streamlit dashboard."""
    app_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    env = {"CONFIG_PATH": str(Path(config_path).resolve())}
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
        env={**dict(__import__("os").environ), **env},
    )


def cmd_visualize(config_path: str, run_id: str | None) -> None:
    """Generate train/test and backtest result figures."""
    from src.visualization.report import generate_report

    config = load_config(config_path)
    result = generate_report(config, run_id=run_id)
    logger.info("Figures written to %s", result.output_dir)


def cmd_pipeline(config_path: str) -> None:
    """Run the full end-to-end pipeline."""
    logger.info("Starting full pipeline")
    cmd_download(config_path)
    cmd_features(config_path)
    cmd_train(config_path, model_type=None)
    cmd_backtest(config_path)
    logger.info("Pipeline complete")


def main() -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(level=args.log_level)
    ensure_directories()

    commands = {
        "download": lambda: cmd_download(args.config, force=getattr(args, "force", False)),
        "features": lambda: cmd_features(args.config),
        "train": lambda: cmd_train(args.config, args.model),
        "backtest": lambda: cmd_backtest(args.config),
        "calibrate": lambda: cmd_calibrate(args.config),
        "paper-trade": lambda: cmd_paper_trade(args.config, args.dry_run),
        "dashboard": lambda: cmd_dashboard(args.config),
        "pipeline": lambda: cmd_pipeline(args.config),
        "visualize": lambda: cmd_visualize(args.config, args.run_id),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
