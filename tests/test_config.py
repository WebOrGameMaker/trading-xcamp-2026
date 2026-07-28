"""Tests for configuration loading."""

from pathlib import Path

import pytest

from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


def test_load_default_config() -> None:
    """Default config loads from configs/default.yaml."""
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    assert config.labels.horizon_days == 5
    assert config.labels.mode == "cross_sectional"
    assert config.labels.positive_quantile == 0.20
    assert config.model.scope == "pooled"
    assert config.model.include_ticker is False
    assert config.model.type == "xgboost"
    assert config.model.task == "regression"
    assert config.strategy.max_positions == 10
    assert config.data.start_date == "2010-01-01"
    assert config.data.train_end_date == "2022-12-31"
    assert config.data.val_start_date == "2023-01-01"
    assert config.data.val_end_date == "2024-12-31"
    assert config.data.test_start_date == "2025-01-01"


def test_config_paths_resolve() -> None:
    """Project paths resolve to existing directories."""
    from src.utils.paths import CONFIG_DIR, DATA_DIR, MODEL_DIR

    assert CONFIG_DIR.exists()
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.name == "data"
    assert MODEL_DIR.name == "models"
