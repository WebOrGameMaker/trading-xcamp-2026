"""Configuration loading from YAML files and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.paths import CONFIG_DIR, PROJECT_ROOT


@dataclass
class DataConfig:
    """Data ingestion settings."""

    start_date: str = "2010-01-01"
    end_date: str | None = None
    train_end_date: str = "2022-12-31"
    val_start_date: str = "2023-01-01"
    val_end_date: str = "2024-12-31"
    test_start_date: str = "2025-01-01"
    timeframe: str = "1Day"
    min_bars: int = 252
    universe_file: str = "sp100_tickers.yaml"


@dataclass
class LabelsConfig:
    """Label generation settings."""

    horizon_days: int = 5
    mode: str = "cross_sectional"
    positive_quantile: float = 0.20
    min_cross_section: int = 10
    threshold: float = 0.0


@dataclass
class FeaturesConfig:
    """Feature engineering settings."""

    indicator_set: str = "standard"


@dataclass
class ModelConfig:
    """Model training settings."""

    scope: str = "pooled"
    type: str = "xgboost"
    include_ticker: bool = False
    walk_forward: bool = False
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    hyperparams: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Trading strategy settings."""

    mode: str = "long_short"
    long_positions: int = 10
    short_positions: int = 10
    long_gross_exposure: float = 0.50
    short_gross_exposure: float = 0.50
    max_weight_per_symbol: float = 0.10
    rebalance_frequency: str = "weekly"
    # Legacy long-only, threshold-gated mode (mode="long_only").
    entry_threshold: float = 0.55
    max_positions: int = 10
    # Confidence-filtered long/short mode (mode="long_short_confidence").
    long_entry_threshold: float = 0.60
    short_entry_threshold: float = 0.40
    # Which prediction column to trade on.
    probability_column: str = "probability"


@dataclass
class BacktestConfig:
    """Backtesting settings."""

    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    benchmark_symbol: str = "SPY"


@dataclass
class ExecutionConfig:
    """Paper trading execution settings."""

    dry_run: bool = True
    paper: bool = True
    max_order_value: float = 10_000.0


@dataclass
class AlpacaConfig:
    """Alpaca API credentials loaded from environment."""

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"


@dataclass
class AppConfig:
    """Top-level application configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> None:
    """Recursively merge dictionary values into a dataclass instance."""
    for key, value in data.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load application configuration from YAML and environment variables.

    Args:
        config_path: Path to YAML config file. Defaults to configs/default.yaml
            or CONFIG_PATH env var.

    Returns:
        Populated AppConfig instance.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    if config_path is None:
        env_path = os.getenv("CONFIG_PATH")
        config_path = Path(env_path) if env_path else CONFIG_DIR / "default.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

    config = AppConfig()

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        _merge_dataclass(config, raw)

    config.alpaca.api_key = os.getenv("ALPACA_API_KEY", config.alpaca.api_key)
    config.alpaca.secret_key = os.getenv("ALPACA_SECRET_KEY", config.alpaca.secret_key)
    config.alpaca.base_url = os.getenv("ALPACA_BASE_URL", config.alpaca.base_url)

    return config
